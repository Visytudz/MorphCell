"""Vector-neuron layers used by the CytoDL point-cloud baseline."""

import torch
from torch import nn

EPS = 1e-12


class VNLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.transpose(1, -1)).transpose(1, -1)


class VNBatchNorm(nn.Module):
    def __init__(self, num_features: int, dim: int):
        super().__init__()
        if dim in (3, 4):
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
        else:
            raise ValueError(f"Unsupported VNBatchNorm dim: {dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        return x / norm.unsqueeze(2) * norm_bn.unsqueeze(2)


class VNLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        out_channels = 1 if share_nonlinearity else in_channels
        self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        return self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )


class VNLinearLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 5,
        share_nonlinearity: bool = False,
        use_batchnorm: bool = True,
        negative_slope: float = 0.2,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.eps = eps
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope
        self.map_to_feat = VNLinear(in_channels, out_channels)
        if use_batchnorm:
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        self.map_to_dir = VNLinear(
            in_channels, 1 if share_nonlinearity else out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.map_to_feat(x)
        if self.use_batchnorm:
            p = self.batchnorm(p)

        d = self.map_to_dir(x)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        return self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + self.eps)) * d)
        )


class VNRotationMatrix(nn.Module):
    def __init__(
        self,
        in_channels: int,
        share_nonlinearity: bool = False,
        use_batchnorm: bool = False,
        eps: float = 1e-12,
        dim: int = 4,
        return_rotated: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.return_rotated = return_rotated
        self.vn1 = VNLinearLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            use_batchnorm=use_batchnorm,
            eps=eps,
        )
        self.vn2 = VNLinearLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            use_batchnorm=use_batchnorm,
            eps=eps,
        )
        self.vn_lin = VNLinear(in_channels // 4, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        z = self.vn1(x)
        z = self.vn2(z)
        z = self.vn_lin(z)

        v1 = z[:, 0, :]
        u1 = v1 / (torch.sqrt((v1 * v1).sum(1, keepdims=True)) + self.eps)

        v2 = z[:, 1, :]
        v2 = v2 - (v2 * u1).sum(1, keepdim=True) * u1
        u2 = v2 / (torch.sqrt((v2 * v2).sum(1, keepdims=True)) + self.eps)

        u3 = torch.cross(u1, u2, dim=1)
        rot = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)

        if not self.return_rotated:
            return rot
        if self.dim == 4:
            x_std = torch.einsum("bijm,bjkm->bikm", x, rot)
        elif self.dim == 3:
            x_std = torch.einsum("bij,bjk->bik", x, rot)
        elif self.dim == 5:
            x_std = torch.einsum("bijmn,bjkmn->bikmn", x, rot)
        else:
            raise ValueError(f"Unsupported VNRotationMatrix dim: {self.dim}")
        return x_std, rot
