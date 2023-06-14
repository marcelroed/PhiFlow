import functools
from typing import Callable
from dataclasses import dataclass

import torch

from phi.field import CenteredGrid
from phi.geom import LevelSet
import phi.math as math


def _broadcast_stack(a, b, dim: int = 0):
    return torch.stack(torch.broadcast_tensors(a, b), dim=dim)


def sign2(x: torch.Tensor):
    # Always returns 1 or -1
    return torch.where(x > 0, 1, -1)



@dataclass
class EdgeCrossings:
    i_intersections: torch.Tensor
    j_intersections: torch.Tensor


stack = _broadcast_stack
n = n_cells_i = 100
m = n_cells_j = 100


def stacki_idx(x, add: int = 0, n: int = n, m: int = m):
    return stack(x + torch.arange(n, device=x.device)[:, None], torch.arange(m, device=x.device)[None, :] + add, dim=-1)

def stackidx_j(x, add: int = 0, n: int = n, m: int = m):
    return stack(torch.arange(n, device=x.device)[:, None] + add, x + torch.arange(m, device=x.device)[None, :], dim=-1)

# @torch.compile
def find_edge_crossings(corner_values: torch.Tensor):
    dev = corner_values.device
    """Find the edges that cross the level set and return a sparse tensor with the differences."""
    # TODO(marcelroed): Make this sparse!
    # We intersect if the signs differ or if the value is zero (then we intersect with the start of the edge)
    # n, m = (d - 1 for d in corner_values.shape)
    n, m = corner_values.shape
    n -= 1
    m -= 1

    i_intersect_kind = sign2(corner_values).diff(dim=0) / 2
    j_intersect_kind = sign2(corner_values).diff(dim=1) / 2

    i_intersect_kind_left = i_intersect_kind[:, :-1]
    i_intersect_kind_right = i_intersect_kind[:, 1:]
    j_intersect_kind_top = j_intersect_kind[:-1, :]
    j_intersect_kind_bot = j_intersect_kind[1:, :]

    i_intersects = (i_intersect_kind != 0) | (corner_values[:-1, :] == 0.)
    j_intersects = (j_intersect_kind != 0) | (corner_values[:, :-1] == 0.)

    i_top_val, i_bot_val = corner_values[:-1, :] * i_intersects, corner_values[1:, :] * i_intersects
    j_left_val, j_right_val = corner_values[:, :-1], corner_values[:, 1:]

    i_intersections = i_top_val / (i_top_val - i_bot_val)
    i_intersections = torch.where(i_intersects, i_intersections, float('nan'))  # (n - 1, m)

    j_intersections = j_left_val / (j_left_val - j_right_val)
    j_intersections = torch.where(j_intersects, j_intersections, float('nan'))  # (n, m - 1)

    i_intersect_left, i_intersect_right = i_intersections[:, :-1], i_intersections[:, 1:]
    j_intersect_top, j_intersect_bot = j_intersections[:-1, :], j_intersections[1:, :]


    starting_point = torch.full(i_intersect_right.shape + (2,), torch.tensor(torch.nan), device=dev)
    starting_point = torch.where(i_intersect_kind_right[..., None] == -1, stacki_idx(i_intersect_right, add=1, n=n, m=m), starting_point)
    starting_point = torch.where(j_intersect_kind_top[..., None] == -1, stackidx_j(j_intersect_top, n=n, m=m), starting_point)
    starting_point = torch.where(j_intersect_kind_bot[..., None] == 1, stackidx_j(j_intersect_bot, add=1, n=n, m=m), starting_point)
    starting_point = torch.where(i_intersect_kind_left[..., None] == 1, stacki_idx(i_intersect_left, n=n, m=m), starting_point)

    ending_point = torch.full(i_intersect_right.shape + (2, ), torch.tensor(torch.nan), device=dev)
    ending_point = torch.where(i_intersect_kind_right[..., None] == 1, stacki_idx(i_intersect_right, add=1, n=n, m=m), ending_point)
    ending_point = torch.where(j_intersect_kind_top[..., None] == 1, stackidx_j(j_intersect_top, n=n, m=m), ending_point)
    ending_point = torch.where(j_intersect_kind_bot[..., None] == -1, stackidx_j(j_intersect_bot, add=1, n=n, m=m), ending_point)
    ending_point = torch.where(i_intersect_kind_left[..., None] == -1, stacki_idx(i_intersect_left, n=n, m=m), ending_point)

    lines = torch.concatenate((starting_point, ending_point), dim=-1)

    return lines
    # return EdgeCrossing(i_intersections=i_intersections, j_intersections=j_intersections)


def march_2d(corner_values: torch.Tensor):
    lines = find_edge_crossings(corner_values)
    lines = lines[~torch.isnan(lines).any(dim=-1)]

    starting_point, ending_point = lines[..., :2], lines[..., 2:]

    line_center = (starting_point + ending_point) / 2
    line_vec = ending_point - starting_point
    area_normal = torch.stack((line_vec[:, 1], -line_vec[:, 0]), dim=-1)

    return line_center, area_normal



def compute_lines(edge_crossings: EdgeCrossings):
    # Compute the linear interpolation guess for zero crossing location
    i_guesses = edge_crossings.i_intersections  # (n - 1, m)
    j_guesses = edge_crossings.j_intersections  # (n, m - 1)

    # TODO(marcelroed): Improve guesses

# backends: ['aot_ts_nvfuser', 'cudagraphs', 'inductor', 'ipex', 'nvprims_nvfuser', 'onnxrt', 'tvm']
def make_march_2d(function: Callable):
    @torch.compile
    def march_2d(corner_positions: torch.Tensor):
        # Compute level set values for all corner positions
        corner_values = function(corner_positions)

        # Find all edges that cross the level set
        lines = find_edge_crossings(corner_values)
        return lines

    @torch.compile
    def wrapper(corner_positions: torch.Tensor, masked: bool = True):
        with_nans = march_2d(corner_positions)
        if not masked:
            return with_nans
        return with_nans[~torch.isnan(with_nans).any(dim=-1)]

    return wrapper
        # Compute the linear interpolation guess for zero crossing location

        # Compute the gradient at the guess in the direction of the edge


def make_march_2d_pressure(function: Callable):
    # @torch.compile()
    def march_2d_pressure(corner_positions, pressure_field, centroid):
        # corner_positions: (n, m, 2)
        # pressure_field: (n, m)
        # centroid: (2,)
        corner_values = function(corner_positions)
        # TODO: Use the gradient function
        # gradient_function = torch.func.grad(function)






def level_set_march(level_set: LevelSet, centered_grid: CenteredGrid):
    corner_positions = centered_grid.element_corners().center
    corner_values = level_set.function(corner_positions)
    corner_values_torch = corner_values.native(corner_values.shape)
    line_centers, line_area_normals = march_2d(corner_values_torch)

    n_lines = line_centers.shape[0]

    return (
        math.tensor(line_centers, math.instance(line=n_lines), math.channel(vector='x,y')),
        math.tensor(line_area_normals, math.instance(line=n_lines), math.channel(vector='x,y'))
    )

def level_set_march_pressure(level_set: LevelSet, pressure_field: CenteredGrid, centroid: math.Tensor):
    corner_positions = pressure_field.element_corners().center
    corner_values = level_set.function(corner_positions)
    corner_values_torch = corner_values.native(corner_values.shape)
    centroid = centroid.native(centroid.shape)

    march_pressure_integral = make_march_2d_pressure()

if __name__ == '__main__':
    import torch.functional as F
    import numpy as np
    from math import sqrt
    from torch.utils.benchmark import Timer
    import torch._dynamo

    dev = 'cuda:2'

    import matplotlib.pyplot as plt

    corner_positions = _broadcast_stack(torch.arange(n_cells_i + 1)[:, None], torch.arange(n_cells_j + 1)[None, :],
                                        dim=-1).float().to(dev)
    # corner_positions = torch.tensor([[[0, 0], [1, 0]], [[0, 1], [1, 1]]], dtype=torch.float32)
    # n_cells_i = corner_positions.shape[0] - 1
    # n_cells_j = corner_positions.shape[1] - 1
    normal = torch.tensor([1, 1 / 2], dtype=torch.float32, device=dev)

    di = 1.0
    dj = 1.0
    corner_positions[..., 0] *= di
    corner_positions[..., 1] *= dj

    bound_limit = torch.max(corner_positions)
    print(bound_limit)


    def plane(x):
        return torch.dot(x, normal) - 0.33


    center = (bound_limit / 2.0).to(dev)
    radius = (torch.linalg.vector_norm(bound_limit) / 4).to(dev)


    def sphere(x: torch.Tensor, center=center, radius=radius):
        return torch.linalg.vector_norm(x - center, dim=-1) - radius


    corner_vals = sphere(corner_positions)
    # print(corner_vals)

    march_2d_sphere = make_march_2d(sphere)

    # print(torch._dynamo.list_backends())
    function_grad = torch.func.grad(sphere)
    # torch._dynamo.config.suppress_errors = True
    # torch._dynamo.config.verbose = True
    corner_positions = corner_positions.to(dev)
    # lines: np.ndarray = march_2d_sphere(corner_positions).cpu().numpy()
    # lines = lines[~np.any(np.isnan(lines), axis=2)]
    # print(lines)
    # start, end = lines[:, 0:2], lines[:, 2:4]
    # startvals = torch.vmap(sphere)(torch.tensor(start, device=dev))
    # print(startvals)
    # vec = end - start
    # for s, v in zip(start, vec):
    #     plt.arrow(*s, *v)
    # # plt.imshow(sphere(corner_positions).numpy())
    # plt.show()

    # Time the function
    print(corner_positions.device)
    res = march_2d_sphere(corner_positions)
    print(res)
    timer = Timer('march_2d_sphere(corner_positions)', globals=globals())
    results = timer.timeit(1000)
    print(results)
