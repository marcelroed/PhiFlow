import warnings
from functools import cache
from typing import Dict, Tuple, Union, List, Callable, Optional

import numpy as np

import phi.field
from phi import math
from . import Box
from ._geom import Geometry, _keep_vector, LineSegment
from ..math import wrap, INF, Shape, channel, spatial, copy_with, Tensor, extrapolation
from ..math._shape import parse_dim_order
from ..math.magic import slicing_dict

# Rotation matrices are of dim ('vector', 'vector_in')

def vector_notation(n: int):
    assert n in (2, 3), 'Only 2D and 3D is currently supported'
    return 'x,y' if n == 2 else 'x,y,z'

def identity(n: int) -> Tensor:
    assert n in (2, 3), 'Only 2D and 3D identity matrices are supported.'
    return math.tensor([[1. if i == j else 0. for i in range(n)] for j in range(n)],
                       channel(vector=vector_notation(n)), channel(vector_in=vector_notation(n)))


def rotate_matmul(left: Tensor, right: Tensor):
    # ik,kj->ij
    # left_in = left._with_shape_replaced(channel(vector_out='x,y', vector_mid='x,y'))
    # right_in = right._with_shape_replaced(channel(vector_mid='x,y', vector_in='x,y'))
    # return math.sum(left_in * right_in, dim='vector_mid')
    return math.dot(left, 'vector_in', right, 'vector')


def rotate_vec(rotation: Tensor, vector: Tensor):
    # left_in = rotation._with_shape_replaced(channel(vector='x,y', vector_mid='x,y'))
    # right_in = vector._with_shape_replaced(vector.shape.non_channel & channel(vector_mid='x,y'))
    return math.dot(rotation, 'vector_in', vector, 'vector')


def cast_rotation(rotation: Tensor | float):
    if isinstance(rotation, float) or rotation.shape.sizes == ():
        return math.tensor([[math.cos(rotation), -math.sin(rotation)], [math.sin(rotation), math.cos(rotation)]], channel(vector='x,y'), channel(vector_in='x,y'))
    else:
        return rotation


class LevelSetTransform:
    def __init__(self, translation: Optional[Tensor] = None, rotation: Optional[Tensor] = None,
                 shape: Optional[Shape] = None):
        if translation is None and shape is None:
            raise ValueError('Either translation or the shape of a translation must be specified.')
        self.shape = translation.shape if shape is None else shape
        self.translation = translation
        self.rotation = rotation if rotation is not None else identity(shape.sizes[0])

    def __call__(self, location: Tensor) -> Tensor:
        if self.translation is not None:
            location = location + self.translation
        if self.rotation is not None:
            # item_names = ','.join(location.shape.channel.item_names[0])
            location = math.dot(self.rotation, 'vector_in', location, 'vector')
        return location

    def inverse(self) -> 'LevelSetTransform':
        if self.translation is None and self.rotation is None:
            return self
        else:
            return LevelSetTransform(-self.translation, -self.rotation)

    def transform(self, translation: Optional[Tensor], rotation: Optional[Tensor] = None) -> 'LevelSetTransform':
        if translation is None:
            translation = self.translation
        elif translation is not None and self.translation is not None:
            translation = translation + self.translation

        if rotation is None:
            rotation = self.rotation
        elif rotation is not None and self.rotation is not None:
            rotation = cast_rotation(rotation)
            assert rotation.shape.sizes == self.rotation.shape.sizes
            rotation = rotation * self.rotation

        return LevelSetTransform(translation=translation,
                                 rotation=rotation,
                                 shape=self.shape)

    def __repr__(self):
        return f'LevelSetTransform(translation={self.translation}, rotation={self.rotation})'


class LevelSet(Geometry):
    @staticmethod
    def bake_properties(function: Callable, evaluation_grid: 'phi.field.CenteredGrid'):
        n = evaluation_grid.shape.spatial_rank
        present = math.cast(function(evaluation_grid.elements.center) >= 0, dtype=evaluation_grid.values.dtype)
        centroid = math.sum(evaluation_grid.elements.center * present, dim=vector_notation(n)) / math.sum(present, dim=vector_notation(n))

        volume = math.sum(present, dim='x,y') * evaluation_grid.elements.volume

        # max_pos = math.max(evaluation_grid.elements.center * present, dim='spatial')
        # min_pos = math.min(evaluation_grid.elements.center * present, dim='spatial')

        # TODO(marcelroed): compute inertia tensor in 3D

        return centroid, volume

    @property
    def center_of_mass(self):
        return self._centroid

    def __init__(self, function: Callable, bounds: Box, *, transforms: Optional[LevelSetTransform] = None, center=None, volume=None):
        self._bounds = bounds
        if bounds.shape.size == 2:
            self._evaluation_grid = phi.field.CenteredGrid(0, x=100, y=100, bounds=bounds, extrapolation=extrapolation.BOUNDARY)
        elif bounds.shape.size == 3:
            self._evaluation_grid = phi.field.CenteredGrid(0, x=100, y=100, z=100, bounds=bounds, extrapolation=extrapolation.BOUNDARY)
        else:
            raise ValueError(f'Bounds must be 2D or 3D, but got {bounds.shape}')

        self._function = function
        self._transforms = LevelSetTransform(shape=bounds.shape) if transforms is None else transforms
        if center is None or volume is None:
            self._centroid, self._volume = LevelSet.bake_properties(self.function, self._evaluation_grid)
        else:
            self._centroid = center
            self._volume = volume
        self._shape = self._centroid.shape
        self._marcher = None

    def function(self, x):
        print(self._transforms)
        x = self._transforms(x)
        return self._function(x)

    @property
    def center(self) -> Tensor:
        return self._centroid  # TODO: Should be center

    @property
    def shape(self) -> Shape:
        # TODO(marcelroed): This should be the shape of the parameters of the level set
        return self._shape

    @property
    def volume(self) -> Tensor:
        # Discretize then compute
        return self._volume

    @property
    def shape_type(self) -> Tensor:
        return math.tensor('L')

    def lies_inside(self, location: Tensor) -> Tensor:
        return self.function(location) < 0

    def approximate_signed_distance(self, location: Union[Tensor, tuple]) -> Tensor:
        return self.function(location)

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        if outward:
            return positions - shift_amount * math.sign(self.function(positions)) * math.to_float(outward)
        else:
            return positions + shift_amount * math.sign(self.function(positions)) * math.to_float(outward)

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError()

    @cache
    def bounding_radius(self) -> Tensor:
        return self._bounds.bounding_radius()

    def bounding_half_extent(self) -> Tensor:
        # It is okay to overshoot this value by just returning the bounding_radius
        return self.bounding_radius()

    def transformed(self, translation: Optional[Tensor] = None, rotation: Optional[Tensor] = None) -> 'LevelSet':
        new_transforms = self._transforms.transform(translation, rotation)
        return LevelSet(self._function, bounds=self._bounds, center=self.center, volume=self.volume, transforms=new_transforms)

    def shifted(self, delta: Tensor) -> 'LevelSet':
        print('Shifting the level set')
        return self.transformed(translation=delta)

    def rotated(self, angle: Union[float, Tensor]) -> 'LevelSet':
        return self.transformed(rotation=angle)

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError()

    def __hash__(self):
        return hash(self._function)

    def __getitem__(self, item):
        return self
        # item = slicing_dict(self, item)
        # return LevelSet(self._center[_keep_vector(item)], self._radius[item])

    @property
    def bounds(self) -> Box:
        return self._bounds


    def marching_cubes(self, corner_positions=None):
        import warp as wp
        import torch
        if corner_positions is None:
            corner_positions = self._evaluation_grid.element_corners().center

        corner_values = self.function(corner_positions)
        corner_values_native: torch.Tensor = corner_values.native(corner_values.shape)

        if self._marcher is None:
            nx, ny, nz = spatial(corner_positions).sizes
            self._marcher = wp.MarchingCubes(nx=nx, ny=ny, nz=nz, max_verts=100_000, max_tris=100_000,
                                             device=str(corner_values_native.device))

        def _marching_cubes(corner_values: Tensor):
            w = wp.from_torch(corner_values)
            # with wp.ScopedStream(torch_stream):
            self._marcher.surface(w, 0.0)
            # wp.synchronize_stream(torch_stream)
            # print(marcher.verts.size, marcher.indices.size)
            verts, tris = wp.to_torch(self._marcher.verts)[:self._marcher.verts.size, :], wp.to_torch(self._marcher.indices)[
                                                                              :self._marcher.indices.size].reshape(-1, 3)
            # wp.stream_from_torch()
            # print(marcher.)
            return verts, tris


        return _marching_cubes(corner_values_native)

