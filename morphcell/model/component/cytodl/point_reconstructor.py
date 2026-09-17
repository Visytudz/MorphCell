"""CytoDL-style point-cloud autoencoder baseline."""

import torch
from torch import nn

from morphcell.model.component.cytodl.dgcnn import CytoDLDGCNN
from morphcell.model.component.cytodl.folding_decoder import CytoDLFoldingDecoder


class CytoDLPointReconstructor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        num_output_points: int = 2025,
        hidden_dim: int = 64,
        hidden_conv2d_channels: list[int] | None = None,
        hidden_conv1d_channels: list[int] | None = None,
        hidden_decoder_dim: int = 512,
        k: int = 20,
        mode: str = "vector",
        include_cross: bool = True,
        include_coords: bool = True,
        symmetry_breaking_axis: int | None = None,
        grid_shape: str = "plane",
        grid_std: float = 0.3,
        sphere_path: str | None = None,
        gaussian_path: str | None = None,
    ):
        super().__init__()
        self.encoder = CytoDLDGCNN(
            num_features=latent_dim,
            hidden_dim=hidden_dim,
            k=k,
            mode=mode,
            hidden_conv2d_channels=hidden_conv2d_channels,
            hidden_conv1d_channels=hidden_conv1d_channels,
            include_cross=include_cross,
            include_coords=include_coords,
            symmetry_breaking_axis=symmetry_breaking_axis,
            x_label="points",
        )
        self.decoder = CytoDLFoldingDecoder(
            input_dim=latent_dim,
            num_output_points=num_output_points,
            hidden_dim=hidden_decoder_dim,
            std=grid_std,
            shape=grid_shape,
            sphere_path=sphere_path,
            gaussian_path=gaussian_path,
            num_coords=3,
        )

    def encode(self, point_cloud: torch.Tensor, get_rotation: bool = False):
        encoded = self.encoder(point_cloud, get_rotation=get_rotation)
        return encoded if get_rotation else encoded["points"]

    def decode(
        self,
        latent: torch.Tensor,
        rotation: torch.Tensor | None = None,
        return_canonical: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        canonical = self.decoder(latent)
        if rotation is None or rotation.ndim != 3:
            reconstruction = canonical
        else:
            reconstruction = torch.einsum("bnj,bjk->bnk", canonical, rotation)

        if return_canonical:
            return {
                "reconstruction": reconstruction,
                "canonical": canonical,
            }
        return reconstruction

    def forward(
        self,
        point_cloud: torch.Tensor,
        return_canonical: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        encoded = self.encode(point_cloud, get_rotation=True)
        return self.decode(
            encoded["points"],
            rotation=encoded["rotation"],
            return_canonical=return_canonical,
        )
