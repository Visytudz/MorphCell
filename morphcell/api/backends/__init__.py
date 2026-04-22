"""Backend registry for model-agnostic inference."""

from morphcell.api.backends.base import InferenceBackend
from morphcell.api.backends.baseline import BaselineBackend
from morphcell.api.backends.pqae import PQAEBackend

__all__ = [
    "InferenceBackend",
    "BaselineBackend",
    "PQAEBackend",
    "create_backend",
]


def create_backend(model, device) -> InferenceBackend:
    from morphcell.model import PQAEPretrain, BaselinePretrain

    if isinstance(model, PQAEPretrain):
        return PQAEBackend(model, device)

    if isinstance(model, BaselinePretrain):
        return BaselineBackend(model)

    raise ValueError(
        f"Unsupported model type: {type(model).__name__}. "
        "Register a backend in morphcell.api.backends.create_backend()."
    )
