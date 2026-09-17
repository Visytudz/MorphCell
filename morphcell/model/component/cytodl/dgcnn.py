"""CytoDL-style DGCNN encoder with optional vector-neuron rotation invariance."""

import torch
from torch import nn

from morphcell.model.component.cytodl.graph_functions import get_graph_features
from morphcell.model.component.cytodl.vnn import (
    VNLeakyReLU,
    VNLinear,
    VNLinearLeakyReLU,
    VNRotationMatrix,
)


def _make_conv(
    in_features: int,
    out_features: int,
    mode: str = "scalar",
    scale_in: int = 1,
    add_in: int = 0,
    include_symmetry: int = 0,
    scale_out: int = 1,
    final: bool = False,
) -> nn.Module:
    in_features = in_features * scale_in + include_symmetry + add_in
    out_features = out_features * scale_out

    if mode == "vector":
        return VNLinearLeakyReLU(
            in_features,
            out_features,
            use_batchnorm=False,
            negative_slope=0,
            dim=(4 if final else 5),
        )
    if mode == "vector2":
        return nn.Sequential(
            VNLeakyReLU(in_features, negative_slope=0.0, share_nonlinearity=False),
            VNLinear(in_features, out_features),
        )

    conv = nn.Conv1d if final else nn.Conv2d
    batch_norm = nn.BatchNorm1d if final else nn.BatchNorm2d
    return nn.Sequential(
        conv(in_features, out_features, kernel_size=1, bias=False),
        batch_norm(out_features),
        nn.LeakyReLU(negative_slope=0.2),
    )


def maxpool(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def meanpool(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return x.mean(dim=dim, keepdim=keepdim)


class CytoDLDGCNN(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        k: int = 20,
        mode: str = "vector",
        hidden_conv2d_channels: list[int] | None = None,
        hidden_conv1d_channels: list[int] | None = None,
        scalar_inds: int | None = None,
        include_cross: bool = True,
        include_coords: bool = True,
        symmetry_breaking_axis: int | None = None,
        collate_intermediates: bool = True,
        x_label: str = "points",
    ):
        super().__init__()
        if mode not in ("scalar", "vector"):
            raise ValueError(f"mode must be 'scalar' or 'vector', got {mode!r}")

        hidden_conv2d_channels = hidden_conv2d_channels or [64, 64, 64, 64]
        expected_final_in = (
            hidden_dim * 2 * len(hidden_conv2d_channels)
            if mode == "vector"
            else hidden_dim
            + sum(
                c_2 * scale
                for c_2, scale in zip(
                    hidden_conv2d_channels[1:],
                    [1]
                    + [
                        (i + 1) * 2
                        for i in range(len(hidden_conv2d_channels) - 2)
                    ],
                )
            )
        )
        hidden_conv1d_channels = hidden_conv1d_channels or [
            expected_final_in,
            num_features,
        ]
        if hidden_conv1d_channels[0] != expected_final_in:
            raise ValueError(
                "hidden_conv1d_channels[0] must match the collated encoder "
                f"feature width ({expected_final_in}), got "
                f"{hidden_conv1d_channels[0]}"
            )
        if hidden_conv1d_channels[-1] != num_features:
            raise ValueError(
                "hidden_conv1d_channels[-1] must match num_features "
                f"({num_features}), got {hidden_conv1d_channels[-1]}"
            )

        self.k = k
        self.x_label = x_label
        self.num_features = num_features
        self.include_coords = include_coords
        self.hidden_conv2d_channels = hidden_conv2d_channels
        self.hidden_conv1d_channels = hidden_conv1d_channels
        self.include_cross = include_cross
        self.hidden_dim = hidden_dim
        self.scalar_inds = scalar_inds
        self.mode = mode
        self.symmetry_breaking_axis = symmetry_breaking_axis
        self.collate_intermediates = collate_intermediates

        include_symmetry = 1 if symmetry_breaking_axis is not None else 0
        self.init_features = 1 if mode == "vector" else 3
        scalar_scale = 2 if scalar_inds else 1

        convs = [
            _make_conv(
                self.init_features,
                hidden_dim,
                mode,
                scale_in=(
                    1 + include_coords + (include_cross if mode == "vector" else 0)
                )
                * scalar_scale,
                include_symmetry=include_symmetry,
            )
        ]

        if mode == "vector":
            convs += [
                nn.Sequential(
                    VNLinear(hidden_dim, hidden_dim * 2),
                    VNLeakyReLU(
                        2 * hidden_dim,
                        negative_slope=0.0,
                        share_nonlinearity=False,
                    ),
                    VNLinear(2 * hidden_dim, hidden_dim),
                )
            ]
            scale_in_list = [2 for _ in range(len(hidden_conv2d_channels) - 1)]
            scale_out_list = [1 for _ in range(len(hidden_conv2d_channels) - 1)]
            next_mode = "vector2"
            prev_slice = -1
            self.pool = meanpool
            self.rotation = VNRotationMatrix(num_features, dim=3, return_rotated=True)
            self.embedding_head = VNLinear(num_features, num_features)
        else:
            scale_in_list = [2] + [
                (i + 1) * 2 for i in range(len(hidden_conv2d_channels) - 2)
            ]
            scale_out_list = [1] + [
                (i + 1) * 2 for i in range(len(hidden_conv2d_channels) - 2)
            ]
            next_mode = "scalar"
            prev_slice = -3
            self.pool = maxpool
            self.embedding_head = nn.Linear(hidden_conv1d_channels[-1], num_features)

        for j, (c_1, c_2) in enumerate(
            zip(hidden_conv2d_channels[:-1], hidden_conv2d_channels[1:])
        ):
            convs += [
                _make_conv(
                    c_1,
                    c_2,
                    mode=next_mode,
                    scale_in=scale_in_list[j],
                    scale_out=scale_out_list[j],
                )
            ]

        final_conv = []
        prev_in = 0
        for j, (c_1, c_2) in enumerate(
            zip(hidden_conv1d_channels[:-1], hidden_conv1d_channels[1:])
        ):
            if j == 1:
                final_conv += _make_conv(
                    c_1, c_2, final=True, add_in=prev_in, mode=next_mode
                )
            else:
                final_conv += _make_conv(c_1, c_2, final=True, mode=next_mode)
            prev_in = final_conv[prev_slice].in_channels

        self.convs = nn.ModuleList(convs)
        self.final_conv = nn.ModuleList(final_conv)

    def get_graph_features(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return get_graph_features(
            x,
            k=self.k,
            mode=self.mode,
            scalar_inds=self.scalar_inds,
            include_cross=self.include_cross,
            include_input=(self.mode == "vector" or self.include_coords or idx > 0),
        )

    def concat_axis(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        axis_tensor = torch.zeros(3, device=x.device, dtype=x.dtype)
        axis_tensor[axis] = 1.0
        axis_tensor = axis_tensor.view(1, 1, 3, 1, 1)
        axis_tensor = axis_tensor.expand(x.shape[0], 1, 3, x.shape[-2], x.shape[-1])
        return torch.cat((x, axis_tensor), dim=1)

    def forward(
        self, x: torch.Tensor, get_rotation: bool = False
    ) -> dict[str, torch.Tensor]:
        x = x.transpose(2, 1)  # [B, 3, N]
        intermediate_outs = []

        for idx, conv in enumerate(self.convs):
            if (idx == 0 and self.mode == "vector") or self.mode == "scalar":
                x = self.get_graph_features(x, idx)

            if idx == 0 and self.symmetry_breaking_axis is not None:
                x = self.concat_axis(x, self.symmetry_breaking_axis)

            pre_x = conv(x)
            if len(pre_x.size()) < 5 and self.mode == "vector" and idx > 0:
                if idx < len(self.convs) - 1:
                    x = self.pool(pre_x, dim=-1, keepdim=True).expand(pre_x.size())
                    x = torch.cat([x, pre_x], dim=1)
                else:
                    x = pre_x
            else:
                x = self.pool(pre_x, dim=-1)
            intermediate_outs.append(x)

        if self.collate_intermediates:
            x = torch.cat(intermediate_outs, dim=1)

        pool_ind = 2 if self.mode == "scalar" else 1
        for j, conv in enumerate(self.final_conv):
            x = conv(x)
            if j == pool_ind:
                x = self.pool(x, dim=-1)

        rot = torch.zeros(1, device=x.device, dtype=x.dtype)
        if self.mode == "vector":
            x, rot = self.rotation(x)
            x = self.embedding_head(x)
            x = torch.norm(x, dim=-1)
            rot = rot.mT
        else:
            x = self.embedding_head(x)

        if get_rotation:
            return {self.x_label: x, "rotation": rot}
        return {self.x_label: x}
