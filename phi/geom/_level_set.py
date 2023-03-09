import warnings
from functools import cache
from typing import Dict, Tuple, Union, List, Callable, Optional

import numpy as np

from phi import math
from ._geom import Geometry, _keep_vector, LineSegment
from ..math import wrap, INF, Shape, channel, spatial, copy_with, Tensor, extrapolation
from ..math._shape import parse_dim_order
from ..math.magic import slicing_dict
from phi.field import CenteredGrid


def identity(n: int) -> Tensor:
    return math.tensor([[1. if i == j else 0. for i in range(n)] for j in range(n)], channel(vector_out='x,y'), channel(vector_in='x,y'))


def rotate_matmul(left: Tensor, right: Tensor):
    # ik,kj->ij
    left_in = left._with_shape_replaced(channel(vector_out='x,y', vector_mid='x,y'))
    right_in = right._with_shape_replaced(channel(vector_mid='x,y', vector_in='x,y'))
    return math.sum(left_in * right_in, dim='vector_mid')


class LevelSetTransform:
    def __init__(self, translation: Optional[Tensor] = None, rotation: Optional[Tensor] = None, shape: Optional[Shape] = None):
        if translation is None and shape is None:
            raise ValueError('Either translation or the shape of a translation must be specified.')
        if translation is None:
            translation = math.zeros(channel(vector='x,y'))
        self.shape = translation.shape if shape is None else shape
        self.translation = translation
        self.rotation = rotation if rotation is not None else identity(self.translation.shape.sizes[0])

    def __call__(self, location: Tensor) -> Tensor:
        if self.translation is not None:
            location = location + self.translation
        if self.rotation is not None:
            location_in = location._with_shape_replaced(location.shape.non_channel & channel(vector_in='x,y'))
            location = math.sum(self.rotation * location_in, dim='vector_in')._with_shape_replaced(location.shape.non_channel & channel(vector='x,y'))
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
            assert rotation.shape.sizes == self.rotation.shape.sizes
            rotation = rotation * self.rotation

        return LevelSetTransform(translation=translation,
                                 rotation=rotation,
                                 shape=self.shape)


class LevelSet(Geometry):
    @staticmethod
    def bake_properties(function: Callable, evaluation_grid: CenteredGrid):
        present = math.cast(function(evaluation_grid.elements.center) >= 0, dtype=evaluation_grid.values.dtype)
        centroid = math.sum(evaluation_grid.elements.center * present, dim='x,y') / math.sum(present, dim='x,y')

        volume = math.sum(present, dim='x,y') * evaluation_grid.elements.volume

        # max_pos = math.max(evaluation_grid.elements.center * present, dim='spatial')
        # min_pos = math.min(evaluation_grid.elements.center * present, dim='spatial')

        # TODO(marcelroed): compute inertia tensor in 3D

        return centroid, volume

    @property
    def center_of_mass(self):
        return self._centroid

    def __init__(self, function: Callable, bounds, *, transforms: Optional[LevelSetTransform] = None, center=None, volume=None):
        self._bounds = bounds
        self._evaluation_grid = CenteredGrid(0, x=100, y=100, bounds=bounds, extrapolation=extrapolation.BOUNDARY)

        self._function = function
        self._transforms = LevelSetTransform(shape=(2,)) if transforms is None else transforms
        if center is None or volume is None:
            self._centroid, self._volume = LevelSet.bake_properties(self.function, self._evaluation_grid)
        else:
            self._centroid = center
            self._volume = volume
        self._shape = self._centroid.shape

    def function(self, x):
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
        # Create a masked distance array and get its maximum value
        mask = self.lies_inside()


    def bounding_half_extent(self) -> Tensor:
        # It is okay to overshoot this value by just returning the bounding_radius
        return self.bounding_radius()

    def transformed(self, translation: Optional[Tensor] = None, rotation: Optional[Tensor] = None) -> 'LevelSet':
        new_transforms = self._transforms.transform(translation, rotation)
        return LevelSet(self._function, bounds=self._bounds, center=self.center, volume=self.volume, transforms=new_transforms)

    def shifted(self, delta: Tensor) -> 'LevelSet':
        return self.transformed(translation=delta)

    def rotated(self, angle: Union[float, Tensor]) -> 'LevelSet':
        return self.transformed(rotation=angle)

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError()

    def __hash__(self):
        return hash(self._function)

    def __getitem__(self, item):
        raise NotImplementedError()