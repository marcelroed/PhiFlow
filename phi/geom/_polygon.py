import warnings
from typing import Dict, Tuple, Union, List

import numpy as np

from phi import math
from ._geom import Geometry, _keep_vector, LineSegment
from ..math import wrap, INF, Shape, channel, spatial, copy_with, Tensor
from ..math._shape import parse_dim_order
from ..math.magic import slicing_dict


class Polygon(Geometry):
    def __init__(self, vertices: Union[Tensor, List[List[float]]]):
        assert vertices.shape.rank == 2
        assert vertices.shape[1] == 2
        self._vertices = vertices

    @property
    def center(self) -> Tensor:
        # The center of the bounding box.
        # That is, the average of the minimum and maximum coordinates.
        min_x = math.min(self.vertices[:, 0])
        max_x = math.max(self.vertices[:, 0])
        min_y = math.min(self.vertices[:, 1])
        max_y = math.max(self.vertices[:, 1])
        return math.stack([(max_x + min_x) / 2, (max_y + min_y) / 2], axis=0)

    def center_of_mass(self) -> Tensor:
        raise NotImplementedError()

    @property
    def shape(self) -> Shape:
        return self._vertices.shape

    @property
    def volume(self) -> Tensor:
        raise NotImplementedError()

    @property
    def shape_type(self) -> Tensor:
        return math.tensor('P')

    def lies_inside(self, location: Tensor) -> Tensor:
        raise NotImplementedError()

    def approximate_signed_distance(self, location: Union[Tensor, tuple]) -> Tensor:
        raise NotImplementedError()

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        raise NotImplementedError()

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError()

    def bounding_radius(self) -> Tensor:
        # The largest distance between any vertex and the center of the bounding box.
        center = self.center
        return math.max(math.vec_length(self._vertices - center))

    def bounding_half_extent(self) -> Tensor:
        raise NotImplementedError()

    def shifted(self, delta: Tensor) -> 'Geometry':
        return Polygon(vertices=self._vertices + delta)

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        from ._transform import rotate
        return rotate(self, angle)

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        # Scale around the center as an anchor point
        center = self.center
        return Polygon((self._vertices - center) * factor + center)

    def __hash__(self):
        return hash(self._vertices)

    def __getitem__(self, item):
        pass

    def faces(self) -> 'FaceStack':
        pass