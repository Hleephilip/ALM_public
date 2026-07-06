"""CUDA jump-flood fill for unobserved texture pixels."""

from __future__ import annotations

import torch

_VORONOI_KERNEL = None


def _get_kernel():
    # Import CuPy lazily. PyTorch3D uses torch.linalg during renderer setup, and
    # loading CuPy first can make it resolve an incompatible system cuSOLVER.
    import cupy

    global _VORONOI_KERNEL
    if _VORONOI_KERNEL is None:
        _VORONOI_KERNEL = cupy.RawKernel(
            r"""
            extern "C" __global__
            void voronoi_pass(
                const long long step,
                const long long height,
                const long long width,
                const long long *ping,
                long long *pong
            ) {
                long long idx = blockIdx.x * blockDim.x + threadIdx.x;
                long long stride = blockDim.x * gridDim.x;
                for (long long pixel = idx; pixel < height * width; pixel += stride) {
                    long long offsets[] = {-1, 0, 1};
                    for (int row = 0; row < 3; ++row) {
                        for (int col = 0; col < 3; ++col) {
                            long long dx = (step * offsets[row]) * width;
                            long long dy = step * offsets[col];
                            long long source = pixel + dx + dy;
                            if (source < 0 || source >= height * width || ping[source] == -1)
                                continue;
                            if (pong[pixel] == -1) {
                                pong[pixel] = ping[source];
                                continue;
                            }
                            long long x1 = pixel / width;
                            long long y1 = pixel % width;
                            long long x2 = pong[pixel] / width;
                            long long y2 = pong[pixel] % width;
                            long long x3 = ping[source] / width;
                            long long y3 = ping[source] % width;
                            long long current_distance =
                                (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
                            long long candidate_distance =
                                (x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3);
                            if (candidate_distance < current_distance)
                                pong[pixel] = ping[source];
                        }
                    }
                }
            }
            """,
            "voronoi_pass",
        )
    return cupy, _VORONOI_KERNEL


def voronoi_fill(
    texture: torch.Tensor,
    mask: torch.Tensor,
    *,
    disabled: bool = False,
) -> torch.Tensor:
    """Propagate valid texture pixels to their nearest unobserved neighbors."""

    if disabled:
        return texture
    valid_coordinates = torch.where(mask > 0)
    if valid_coordinates[0].numel() == 0:
        return texture

    height, width, channels = texture.shape
    flat_indices = torch.arange(height * width, device=texture.device, dtype=torch.int64).reshape(height, width)
    index_map = torch.full_like(flat_indices, -1)
    index_map[valid_coordinates] = flat_indices[valid_coordinates]

    cupy, kernel = _get_kernel()
    ping = cupy.asarray(index_map)
    pong = cupy.copy(ping)
    step = max(height, width) // 2
    while step:
        kernel((min(height, 1024),), (min(width, 1024),), (step, height, width, ping, pong))
        ping, pong = pong, ping
        step //= 2

    nearest = torch.as_tensor(ping, device=texture.device)
    filled = torch.index_select(texture.reshape(height * width, channels), 0, nearest.reshape(height * width))
    return filled.reshape(height, width, channels)
