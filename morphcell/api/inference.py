"""Unified inference API."""

from typing import Callable

import numpy as np
import torch

from morphcell.api.backends import create_backend
from morphcell.api.model import InferenceModel
from morphcell.api.types import FeatureBundle, ModelCapabilities
from morphcell.api.utils import prepare_batch_input, prepare_input


class MorphCellInference:
    """Unified inference interface built on symmetric model backends."""

    def __init__(
        self,
        config_dir: str,
        config_name: str,
        checkpoint_path: str,
        device: str = "auto",
        batch_size: int = 32,
    ):
        self._loader = InferenceModel(config_dir, config_name, checkpoint_path, device)
        self.device = self._loader.device
        self.batch_size = batch_size
        self._backend = create_backend(self._loader.get_model(), self.device)

    def extract_features(
        self,
        data: str | np.ndarray | torch.Tensor | list,
        normalize: bool = True,
    ) -> FeatureBundle:
        pts = self._prepare(data, normalize)
        return self._backend.extract_features(pts)

    def extract_global_features(
        self,
        data: str | np.ndarray | torch.Tensor | list,
        normalize: bool = True,
    ) -> torch.Tensor:
        pts = self._prepare(data, normalize)
        return self._backend.extract_global_features(pts)

    def extract_local_features(
        self,
        data: str | np.ndarray | torch.Tensor | list,
        normalize: bool = True,
    ) -> torch.Tensor | None:
        pts = self._prepare(data, normalize)
        return self._backend.extract_local_features(pts)

    @torch.no_grad()
    def reconstruct(
        self,
        data: str | np.ndarray | torch.Tensor | list,
        normalize: bool = True,
        return_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        pts = self._prepare(data, normalize)
        reconstructed = self._backend.reconstruct(pts)
        return self._postprocess(reconstructed, return_numpy)

    @torch.no_grad()
    def reconstruct_from_features(
        self,
        features: FeatureBundle,
        return_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        reconstructed = self._backend.reconstruct_from_features(features)
        squeeze_if_single = features.global_features.ndim == 1
        return self._postprocess(
            reconstructed, return_numpy, squeeze_if_single=squeeze_if_single
        )

    @torch.no_grad()
    def fusion_reconstruct(
        self,
        data_list: list[str | np.ndarray | torch.Tensor],
        weights: list[float] | np.ndarray | torch.Tensor,
        normalize: bool = True,
        normalize_weights: bool = True,
        return_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        if len(data_list) != len(weights):
            raise ValueError(
                f"Length mismatch: data_list has {len(data_list)} items, "
                f"weights has {len(weights)} items"
            )
        if len(data_list) == 0:
            raise ValueError("data_list cannot be empty")

        weights_tensor = self._prepare_weights(weights, normalize_weights)
        pts_list = [prepare_input(data, self.device, normalize) for data in data_list]
        reconstructed = self._backend.fusion_reconstruct(pts_list, weights_tensor)
        return self._postprocess(reconstructed, return_numpy)

    @torch.no_grad()
    def cross_reconstruct(
        self,
        data: str | np.ndarray | torch.Tensor | list,
        normalize: bool = True,
        return_numpy: bool = True,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        pts = self._prepare(data, normalize)
        results = self._backend.cross_reconstruct(pts)
        if return_numpy:
            results = {key: value.detach().cpu().numpy() for key, value in results.items()}
            if list(results.values())[0].shape[0] == 1:
                results = {key: value[0] for key, value in results.items()}
        return results

    def compute_pca_saliency(
        self,
        pts: torch.Tensor | np.ndarray,
        pca_components: np.ndarray,
        aggregate: str = "norm",
    ):
        return self._backend.compute_pca_saliency(pts, pca_components, aggregate)

    def compute_gradient_saliency(
        self,
        pts: torch.Tensor | np.ndarray,
        target_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        aggregate: str = "norm",
    ):
        return self._backend.compute_gradient_saliency(pts, target_fn, aggregate)

    def get_model(self):
        return self._loader.get_model()

    def get_backend(self):
        return self._backend

    def get_device(self) -> torch.device:
        return self.device

    def get_capabilities(self) -> ModelCapabilities:
        return self._backend.capabilities

    def _prepare(
        self,
        data: str | np.ndarray | torch.Tensor | list,
        normalize: bool,
    ) -> torch.Tensor:
        if isinstance(data, list):
            return prepare_batch_input(data, self.device, normalize)
        return prepare_input(data, self.device, normalize)

    def _prepare_weights(
        self,
        weights: list[float] | np.ndarray | torch.Tensor,
        normalize_weights: bool,
    ) -> torch.Tensor:
        if isinstance(weights, list):
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        elif isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float().to(self.device)
        else:
            weights = weights.float().to(self.device)

        if normalize_weights:
            weights = weights / weights.sum()
        return weights

    def _postprocess(
        self,
        tensor: torch.Tensor,
        return_numpy: bool,
        squeeze_if_single: bool = True,
    ) -> np.ndarray | torch.Tensor:
        if not return_numpy:
            return tensor

        result = tensor.detach().cpu().numpy()
        if squeeze_if_single and result.shape[0] == 1:
            result = result[0]
        return result
