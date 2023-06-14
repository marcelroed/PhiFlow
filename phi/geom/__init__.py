"""
Differentiable geometry package.

Classes:

* `Geometry` (base type)
* `Box`
* `Sphere`

See the `phi.geom` module documentation at https://tum-pbs.github.io/PhiFlow/Geometry.html
"""
from ..math import stack, concat, pack_dims  # for compatibility
from ._geom import Geometry, Point, assert_same_rank, invert, LineSegment, subdivide_line_segment, subdivide_line_segment_to_size
from ._union import union
from ._box import Box, GridCell, GridCell as UniformGrid, BaseBox, Cuboid
from ._sphere import Sphere
from ._level_set import LevelSet
from ._transform import embed, infinite_cylinder

__all__ = [key for key in globals().keys() if not key.startswith('_')]
