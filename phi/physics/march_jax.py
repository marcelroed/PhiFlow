import functools
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from timeit import Timer

from matplotlib import pyplot as plt


def sign2(x):
    # Always returns 1 or -1
    return jnp.where(x > 0, 1, -1)


n = n_cells_i = 100
m = n_cells_j = 100


def stack(a, b, axis=0):
    return jnp.stack(jnp.broadcast_arrays(a, b), axis=axis)


def stacki_idx(x, add: int = 0, n: int = n, m: int = m):
    return stack(x + jnp.arange(n)[:, None], jnp.arange(m)[None, :] + add, axis=-1)


def stackidx_j(x, add: int = 0, n: int = n, m: int = m):
    return stack(jnp.arange(n)[:, None] + add, x + jnp.arange(m)[None, :], axis=-1)

def find_edge_crossing_cell(corner_values):
    # Takes a 2x2 array of corner values and returns the start and end point of the line
    return


def find_edge_crossings(corner_values):
    """Find the edges that cross the level set and return a sparse tensor with the differences."""
    # TODO(marcelroed): Make this sparse!
    # We intersect if the signs differ or if the value is zero (then we intersect with the start of the edge)
    # n, m = (d - 1 for d in corner_values.shape)
    i_intersect_kind = jnp.diff(sign2(corner_values), axis=0) / 2
    j_intersect_kind = jnp.diff(sign2(corner_values), axis=1) / 2

    i_intersect_kind_left = i_intersect_kind[:, :-1]
    i_intersect_kind_right = i_intersect_kind[:, 1:]
    j_intersect_kind_top = j_intersect_kind[:-1, :]
    j_intersect_kind_bot = j_intersect_kind[1:, :]

    i_intersects = (i_intersect_kind != 0) | (corner_values[:-1, :] == 0.)
    j_intersects = (j_intersect_kind != 0) | (corner_values[:, :-1] == 0.)

    i_top_val, i_bot_val = corner_values[:-1, :] * i_intersects, corner_values[1:, :] * i_intersects
    j_left_val, j_right_val = corner_values[:, :-1], corner_values[:, 1:]

    i_intersections = i_top_val / (i_top_val - i_bot_val)
    i_intersections = jnp.where(i_intersects, i_intersections, jnp.nan)  # (n - 1, m)

    j_intersections = j_left_val / (j_left_val - j_right_val)
    j_intersections = jnp.where(j_intersects, j_intersections, jnp.nan)  # (n, m - 1)

    i_intersect_left, i_intersect_right = i_intersections[:, :-1], i_intersections[:, 1:]
    j_intersect_top, j_intersect_bot = j_intersections[:-1, :], j_intersections[1:, :]

    starting_point = jnp.full(i_intersect_right.shape + (2,), jnp.nan)
    starting_point = jnp.where(i_intersect_kind_right[..., None] == -1, stacki_idx(i_intersect_right, add=1),
                               starting_point)
    starting_point = jnp.where(j_intersect_kind_top[..., None] == -1, stackidx_j(j_intersect_top), starting_point)
    starting_point = jnp.where(j_intersect_kind_bot[..., None] == 1, stackidx_j(j_intersect_bot, add=1), starting_point)
    starting_point = jnp.where(i_intersect_kind_left[..., None] == 1, stacki_idx(i_intersect_left), starting_point)

    ending_point = jnp.full(i_intersect_right.shape + (2,), jnp.nan)
    ending_point = jnp.where(i_intersect_kind_right[..., None] == 1, stacki_idx(i_intersect_right, add=1), ending_point)
    ending_point = jnp.where(j_intersect_kind_top[..., None] == 1, stackidx_j(j_intersect_top), ending_point)
    ending_point = jnp.where(j_intersect_kind_bot[..., None] == -1, stackidx_j(j_intersect_bot, add=1), ending_point)
    ending_point = jnp.where(i_intersect_kind_left[..., None] == -1, stacki_idx(i_intersect_left), ending_point)

    lines = jnp.concatenate((starting_point, ending_point), axis=-1)

    return lines
    # return EdgeCrossing(i_intersections=i_intersections, j_intersections=j_intersections)


def make_march_2d(function: Callable):
    @jax.jit
    def march_2d(corner_positions):
        print('Tracing!')
        # Compute level set values for all corner positions
        corner_values = function(corner_positions)

        # Find all edges that cross the level set
        lines = find_edge_crossings(corner_values)
        return lines

    @jax.jit
    def nanmask(with_nans):
        mask = ~jnp.isnan(with_nans).any(axis=-1)
        return mask

    @functools.wraps(march_2d)
    def wrapper(corner_positions, masked=True):
        with_nans = march_2d(corner_positions)
        if not masked:
            return np.array(with_nans)
        mask = nanmask(with_nans)
        return jnp.array(np.array(with_nans)[np.array(mask)])

    return wrapper


def _bench_march_2d():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    corner_positions = stack(jnp.arange(n_cells_i + 1)[:, None], jnp.arange(n_cells_j + 1)[None, :],
                             axis=-1).astype(jnp.float32)

    di = 1.0
    dj = 1.0
    corner_positions = corner_positions.at[..., 0].mul(di) \
        .at[..., 1].mul(dj)

    # corner_positions: (n, m, 2)

    bound_limit = jnp.max(corner_positions, axis=(0, 1))

    center = (bound_limit / 2.0)
    radius = jnp.linalg.norm(bound_limit, axis=0) / 4


    def sphere(x):
        return jnp.linalg.norm(x - center, axis=-1) - radius


    march_2d_sphere = make_march_2d(sphere)
    result = march_2d_sphere(corner_positions)
    print(result)

    timer = Timer('march_2d_sphere(corner_positions)', globals=locals())
    n_trials = 100
    results = timer.timeit(n_trials) / n_trials
    print(f'{results * 1000:.4f} ms')

    timer = Timer('march_2d_sphere(corner_positions, masked=False)', globals=locals())
    n_trials = 100
    results = timer.timeit(n_trials) / n_trials
    print(f'{results * 1000:.4f} ms')


if __name__ == '__main__':
    _bench_march_2d()