"""Graph feature helpers for CytoDL-style point-cloud encoders."""

import torch


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1)
    idx_base = idx_base * num_points
    idx = idx + idx_base
    return idx.view(-1)


def get_graph_features(
    x: torch.Tensor,
    k: int = 20,
    idx: torch.Tensor | None = None,
    mode: str = "scalar",
    scalar_inds: int | None = None,
    include_cross: bool = True,
    include_input: bool = True,
) -> torch.Tensor:
    batch_size = x.shape[0]
    num_points = x.shape[-1]
    if len(x.shape) not in (3, 4):
        raise ValueError(f"Expected x with 3 or 4 dims, got shape {tuple(x.shape)}")
    if len(x.shape) == 4 and mode != "vector":
        raise ValueError("4D vector features require mode='vector'")

    if mode == "vector" and len(x.size()) == 3:
        x = x.unsqueeze(1)  # [B, 1, 3, N]

    x = x.view(batch_size, -1, num_points)

    if scalar_inds:
        scal = x[:, scalar_inds - 1 :, :]
        x = x[:, : scalar_inds - 1, :]
        num_scalar_points = scal.size(1)

    if idx is None:
        idx = knn(x, k=k)

    num_dims = x.size(1)
    if mode == "vector":
        num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]

    if mode == "vector":
        feature_view_dims = (batch_size, num_points, k, num_dims, 3)
        x_view_dims = (batch_size, num_points, 1, num_dims, 3)
        repeat_dims = (1, 1, k, 1, 1)
        permute_dims = (0, 3, 4, 1, 2)
    else:
        feature_view_dims = (batch_size, num_points, k, num_dims)
        x_view_dims = (batch_size, num_points, 1, num_dims)
        repeat_dims = (1, 1, k, 1)
        permute_dims = (0, 3, 1, 2)

    feature = feature.view(*feature_view_dims)
    x = x.view(*x_view_dims).repeat(*repeat_dims)

    if mode == "vector" and include_cross:
        cross = torch.cross(feature, x, dim=-1)
        feature = torch.cat((feature - x, cross), dim=3)
    else:
        feature = feature - x

    if include_input:
        feature = torch.cat((feature, x), dim=3)

    feature = feature.permute(*permute_dims).contiguous()

    if scalar_inds:
        feature_unit_vector = feature / torch.norm(feature, dim=1).unsqueeze(dim=1)
        scal = scal.transpose(2, 1).contiguous()
        scal = scal.view(batch_size, num_points, 1, num_scalar_points, 1).repeat(
            1, 1, k, 1, 1
        )
        scal = scal.permute(0, 3, 4, 1, 2).contiguous()
        scal = scal * feature_unit_vector
        feature = torch.cat((feature, scal), dim=1)

    return feature
