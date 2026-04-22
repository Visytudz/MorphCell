"""Shared API types for model-agnostic inference."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class FeatureBundle:
    """Model-agnostic feature container."""

    global_features: torch.Tensor
    local_features: torch.Tensor | None = None
    pooled_features: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCapabilities:
    """Capability flags exposed by an inference backend."""

    global_features: bool = True
    local_features: bool = False
    reconstruct_from_input: bool = True
    reconstruct_from_features: bool = True
    feature_fusion: bool = True
    cross_reconstruction: bool = False
    saliency: bool = False
