"""Abstract base class for inference backends."""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch

from morphcell.api.types import FeatureBundle, ModelCapabilities


class InferenceBackend(ABC):
    """Model-specific backend exposed through a common inference interface."""

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        ...

    @abstractmethod
    def extract_features(self, data: torch.Tensor) -> FeatureBundle:
        ...

    def extract_global_features(self, data: torch.Tensor) -> torch.Tensor:
        return self.extract_features(data).global_features

    def extract_local_features(self, data: torch.Tensor) -> torch.Tensor | None:
        return self.extract_features(data).local_features

    @abstractmethod
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def reconstruct_from_features(self, features: FeatureBundle) -> torch.Tensor:
        ...

    @abstractmethod
    def fuse_features(
        self, features_list: list[FeatureBundle], weights: torch.Tensor
    ) -> FeatureBundle:
        ...

    def fusion_reconstruct(
        self, data_list: list[torch.Tensor], weights: torch.Tensor
    ) -> torch.Tensor:
        features_list = [self.extract_features(data) for data in data_list]
        fused = self.fuse_features(features_list, weights)
        return self.reconstruct_from_features(fused)

    def cross_reconstruct(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support cross reconstruction."
        )

    def compute_pca_saliency(
        self,
        pts: torch.Tensor | np.ndarray,
        pca_components: np.ndarray,
        aggregate: str = "norm",
    ):
        raise NotImplementedError(
            f"{type(self).__name__} does not support PCA saliency."
        )

    def compute_gradient_saliency(
        self,
        pts: torch.Tensor | np.ndarray,
        target_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        aggregate: str = "norm",
    ):
        raise NotImplementedError(
            f"{type(self).__name__} does not support gradient saliency."
        )
