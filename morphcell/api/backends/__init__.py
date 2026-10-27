"""Backend registry for model-agnostic inference."""

from morphcell.api.backends.base import InferenceBackend
from morphcell.api.backends.cytodl_point import CytoDLPointBackend
from morphcell.api.backends.dfn import DFNBackend
from morphcell.api.backends.morphcell import MorphCellBackend

__all__ = [
    "InferenceBackend",
    "CytoDLPointBackend",
    "DFNBackend",
    "MorphCellBackend",
    "create_backend",
]


def create_backend(model, device) -> InferenceBackend:
    from morphcell.model import MorphCellPretrain, DFNPretrain, CytoDLPointPretrain

    if isinstance(model, MorphCellPretrain):
        return MorphCellBackend(model, device)

    if isinstance(model, DFNPretrain):
        return DFNBackend(model)

    if isinstance(model, CytoDLPointPretrain):
        return CytoDLPointBackend(model)

    raise ValueError(
        f"Unsupported model type: {type(model).__name__}. "
        "Register a backend in morphcell.api.backends.create_backend()."
    )
