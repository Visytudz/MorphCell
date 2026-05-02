"""Backend registry for model-agnostic inference."""

from morphcell.api.backends.base import InferenceBackend
from morphcell.api.backends.dfn import DFNBackend
from morphcell.api.backends.pqae import PQAEBackend

__all__ = [
    "InferenceBackend",
    "DFNBackend",
    "PQAEBackend",
    "create_backend",
]


def create_backend(model, device) -> InferenceBackend:
    from morphcell.model import PQAEPretrain, DFNPretrain

    if isinstance(model, PQAEPretrain):
        return PQAEBackend(model, device)

    if isinstance(model, DFNPretrain):
        return DFNBackend(model)

    raise ValueError(
        f"Unsupported model type: {type(model).__name__}. "
        "Register a backend in morphcell.api.backends.create_backend()."
    )
