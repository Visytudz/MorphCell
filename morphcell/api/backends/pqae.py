"""PQAE backend for feature extraction, reconstruction, and saliency."""

from typing import Callable

import numpy as np
import torch

from morphcell.api.backends.base import InferenceBackend
from morphcell.api.types import FeatureBundle, ModelCapabilities
from morphcell.api.utils import prepare_input


class PQAEBackend(InferenceBackend):
    """Inference backend for PQAE models."""

    def __init__(self, model, device: torch.device):
        self.model = model
        self.extractor = model.extractor
        self.view_generator = model.view_generator
        self.device = device

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            global_features=True,
            local_features=True,
            reconstruct_from_input=True,
            reconstruct_from_features=True,
            feature_fusion=True,
            cross_reconstruction=True,
            saliency=True,
        )

    def extract_features(self, data: torch.Tensor) -> FeatureBundle:
        cls_features, patch_features, centers, group = self.extractor(data)
        global_features = cls_features.squeeze(1)
        pooled_patch = patch_features.max(dim=1)[0]
        pooled_features = torch.cat([global_features, pooled_patch], dim=-1)
        return FeatureBundle(
            global_features=global_features,
            local_features=patch_features,
            pooled_features=pooled_features,
            aux={"centers": centers, "group": group},
        )

    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(data)
        return self.reconstruct_from_features(features)

    def reconstruct_from_features(self, features: FeatureBundle) -> torch.Tensor:
        global_features = features.global_features
        local_features = features.local_features

        if global_features.ndim == 1:
            global_features = global_features.unsqueeze(0)
        if global_features.ndim == 2:
            global_features = global_features.unsqueeze(1)

        return self.model.self_reconstruction(global_features, local_features)

    def fuse_features(
        self, features_list: list[FeatureBundle], weights: torch.Tensor
    ) -> FeatureBundle:
        global_features = sum(
            weight * features.global_features
            for weight, features in zip(weights, features_list)
        )

        local_features = None
        if features_list and features_list[0].local_features is not None:
            local_features = sum(
                weight * features.local_features
                for weight, features in zip(weights, features_list)
            )

        pooled_features = None
        if local_features is not None:
            pooled_patch = local_features.max(dim=1)[0]
            pooled_features = torch.cat([global_features, pooled_patch], dim=-1)
        elif features_list and features_list[0].pooled_features is not None:
            pooled_features = sum(
                weight * features.pooled_features
                for weight, features in zip(weights, features_list)
            )

        return FeatureBundle(
            global_features=global_features,
            local_features=local_features,
            pooled_features=pooled_features,
        )

    def cross_reconstruct(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        relative_center_1_2, (view1_rot, view1, _), (view2_rot, view2, _) = (
            self.view_generator(data)
        )

        _, patch_features1, centers1, group1 = self.extractor(view1_rot)
        _, patch_features2, centers2, group2 = self.extractor(view2_rot)

        cross_recon1, cross_recon2 = self.model.cross_reconstruction(
            patch_features1, patch_features2, centers1, centers2, relative_center_1_2
        )

        return {
            "view1": view1,
            "view2": view2,
            "view1_rot": view1_rot,
            "view2_rot": view2_rot,
            "cross_recon1": (cross_recon1 + centers1.unsqueeze(2)).flatten(1, 2),
            "cross_recon2": (cross_recon2 + centers2.unsqueeze(2)).flatten(1, 2),
            "group1": (group1 + centers1.unsqueeze(2)).flatten(1, 2),
            "group2": (group2 + centers2.unsqueeze(2)).flatten(1, 2),
        }

    def compute_pca_saliency(
        self,
        pts: torch.Tensor | np.ndarray,
        pca_components: np.ndarray,
        aggregate: str = "norm",
    ):
        pca_components_tensor = torch.from_numpy(pca_components).float().to(self.device)
        saliency_list = []
        group_pts = None

        for pc_vector in pca_components_tensor:
            def target_fn(cls_feat, patch_feat):
                cls_squeezed = cls_feat.squeeze(1)
                patch_max = patch_feat.max(dim=1)[0]
                combined = torch.cat([cls_squeezed, patch_max], dim=-1)
                return (combined * pc_vector).sum()

            saliency, pts_coords = self.compute_gradient_saliency(
                pts, target_fn, aggregate=aggregate
            )
            saliency_list.append(saliency)
            if group_pts is None:
                group_pts = pts_coords

        return np.array(saliency_list), group_pts

    def compute_gradient_saliency(
        self,
        pts: torch.Tensor | np.ndarray | str,
        target_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        aggregate: str = "norm",
    ):
        if isinstance(pts, str):
            pts = prepare_input(pts, self.device, normalize=True)
        elif isinstance(pts, np.ndarray):
            pts = torch.from_numpy(pts).float().to(self.device)
        else:
            pts = pts.to(self.device)

        squeeze_output = False
        if pts.ndim == 2:
            pts = pts.unsqueeze(0)
            squeeze_output = True

        with torch.no_grad():
            neighborhood, centers = self.extractor.grouping(pts)

        neighborhood_grad = neighborhood.clone().requires_grad_(True)
        tokens = self.extractor.patch_embed(neighborhood_grad)
        cls_feat, patch_feat = self.extractor.encoder(tokens, centers)
        target = target_fn(cls_feat, patch_feat)

        if target.numel() != 1:
            raise ValueError(
                f"target_fn must return a scalar tensor, got shape {target.shape}. "
                "Try adding .sum() to reduce to scalar."
            )

        target.backward()
        grad = neighborhood_grad.grad
        bsz, num_patch, num_group = grad.shape[:3]

        if aggregate == "norm":
            saliency_flat = grad.norm(dim=-1).reshape(bsz, num_patch * num_group)
        elif aggregate == "abs":
            saliency_flat = grad.abs().sum(dim=-1).reshape(bsz, num_patch * num_group)
        elif aggregate == "raw":
            saliency_flat = grad.reshape(bsz, num_patch * num_group, 3)
        else:
            raise ValueError(
                f"Unknown aggregate method: {aggregate}. Use 'norm', 'abs', or 'raw'."
            )

        group_pts_flat = (neighborhood + centers.unsqueeze(2)).reshape(
            bsz, num_patch * num_group, 3
        )

        saliency_np = saliency_flat.detach().cpu().numpy()
        group_pts_np = group_pts_flat.detach().cpu().numpy()

        if squeeze_output:
            saliency_np = saliency_np.squeeze(0)
            group_pts_np = group_pts_np.squeeze(0)

        return saliency_np, group_pts_np
