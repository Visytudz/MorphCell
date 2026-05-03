"""FoldingNet decoder used by the CytoDL point-cloud baseline."""

import numpy as np
import torch
from torch import nn


class CytoDLFoldingDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_output_points: int,
        hidden_dim: int = 512,
        std: float = 0.3,
        shape: str = "plane",
        sphere_path: str | None = None,
        gaussian_path: str | None = None,
        num_coords: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_output_points = num_output_points
        self.shape = shape
        self.num_coords = num_coords

        if shape == "plane":
            self.grid_dim = 2
            grid_side = int(np.sqrt(num_output_points))
            if grid_side * grid_side != num_output_points:
                raise ValueError(
                    "num_output_points must be a perfect square when shape='plane'"
                )
            range_x = torch.linspace(-std, std, grid_side)
            range_y = torch.linspace(-std, std, grid_side)
            x_coor, y_coor = torch.meshgrid(range_x, range_y, indexing="ij")
            grid = torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)
        elif shape == "sphere":
            if not sphere_path:
                raise ValueError("sphere_path is required when shape='sphere'")
            self.grid_dim = 3
            grid = torch.tensor(np.load(sphere_path)).float()
        elif shape == "gaussian":
            if not gaussian_path:
                raise ValueError("gaussian_path is required when shape='gaussian'")
            self.grid_dim = 3
            grid = torch.tensor(np.load(gaussian_path)).float()
        else:
            raise ValueError(f"Unsupported folding source shape: {shape}")

        self.register_buffer("grid", grid)

        self.project = (
            nn.Linear(input_dim, hidden_dim, bias=False)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.folding1 = nn.Sequential(
            nn.Linear(hidden_dim + self.grid_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_coords),
        )
        self.folding2 = nn.Sequential(
            nn.Linear(hidden_dim + num_coords, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_coords),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        grid = self.grid.unsqueeze(0).expand(x.shape[0], -1, -1).type_as(x)
        x = x.unsqueeze(1)
        cw_exp = x.expand(-1, grid.shape[1], -1)
        folding_result1 = self.folding1(torch.cat((cw_exp, grid), dim=2))
        return self.folding2(torch.cat((cw_exp, folding_result1), dim=2))
