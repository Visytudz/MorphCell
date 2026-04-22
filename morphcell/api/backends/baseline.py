"""Baseline backend for encoder-decoder models like FoldingNet and DGCNN."""

import torch

from morphcell.api.backends.base import InferenceBackend
from morphcell.api.types import FeatureBundle, ModelCapabilities


class BaselineBackend(InferenceBackend):
    """Inference backend for global-feature encoder-decoder baselines."""

    def __init__(self, model):
        self.model = model
        self.extractor = model.extractor
        self.decoder = model.decoder
        self.reconstructor = model.reconstructor

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            global_features=True,
            local_features=False,
            reconstruct_from_input=True,
            reconstruct_from_features=True,
            feature_fusion=True,
            cross_reconstruction=False,
            saliency=False,
        )

    def extract_features(self, data: torch.Tensor) -> FeatureBundle:
        codeword = self.extractor(data).squeeze(-1)
        return FeatureBundle(
            global_features=codeword,
            local_features=None,
            pooled_features=codeword,
        )

    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        return self.reconstructor(data)

    def reconstruct_from_features(self, features: FeatureBundle) -> torch.Tensor:
        global_features = features.global_features
        if global_features.ndim == 1:
            global_features = global_features.unsqueeze(0)
        if global_features.ndim == 2:
            global_features = global_features.unsqueeze(-1)
        return self.decoder(global_features)

    def fuse_features(
        self, features_list: list[FeatureBundle], weights: torch.Tensor
    ) -> FeatureBundle:
        global_features = sum(
            weight * features.global_features
            for weight, features in zip(weights, features_list)
        )
        return FeatureBundle(
            global_features=global_features,
            local_features=None,
            pooled_features=global_features,
        )
