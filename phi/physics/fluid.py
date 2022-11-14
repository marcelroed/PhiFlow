"""
Functions for simulating incompressible fluids, both grid-based and particle-based.

The main function for incompressible fluids (Eulerian as well as FLIP / PIC) is `make_incompressible()` which removes the divergence of a velocity field.
"""
from typing import Tuple

from phi import math, field
from phi.math import wrap, channel
from phi.field import SoftGeometryMask, AngularVelocity, Grid, divergence, spatial_gradient, where, CenteredGrid, \
    PointCloud
from phi.geom import union, Geometry
from ..field._embed import FieldEmbedding
from ..field._grid import GridType
from ..math import extrapolation, NUMPY, batch, shape, non_channel, expand
from ..math._magic_ops import copy_with
from ..math.extrapolation import combine_sides, Extrapolation


class Obstacle:
    """
    An obstacle defines boundary conditions inside a geometry.
    It can also have a linear and angular velocity.
    """

    def __init__(self, geometry, velocity=0, angular_velocity=0):
        """
        Args:
            geometry: Physical shape and size of the obstacle.
            velocity: Linear velocity vector of the obstacle.
            angular_velocity: Rotation speed of the obstacle. Scalar value in 2D, vector in 3D.
        """
        self.geometry = geometry
        self.velocity = wrap(velocity, channel(geometry)) if isinstance(velocity, (tuple, list)) else velocity
        self.angular_velocity = angular_velocity
        self.shape = shape(geometry) & non_channel(self.velocity) & non_channel(angular_velocity)

    @property
    def is_stationary(self):
        """ Test whether the obstacle is completely still. """
        return isinstance(self.velocity, (int, float)) and self.velocity == 0 and isinstance(self.angular_velocity, (
            int, float)) and self.angular_velocity == 0

    def copied_with(self, **kwargs):
        geometry, velocity, angular_velocity = self.geometry, self.velocity, self.angular_velocity
        if 'geometry' in kwargs:
            geometry = kwargs['geometry']
        if 'velocity' in kwargs:
            velocity = kwargs['velocity']
        if 'angular_velocity' in kwargs:
            angular_velocity = kwargs['angular_velocity']
        return Obstacle(geometry, velocity, angular_velocity)


def make_incompressible(velocity: GridType,
                        obstacles: tuple or list = (),
                        solve=math.Solve('auto', 1e-5, 1e-5, gradient_solve=math.Solve('auto', 1e-5, 1e-5)),
                        active: CenteredGrid = None) -> Tuple[GridType, CenteredGrid]:
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.
    
    This method is similar to :func:`field.divergence_free()` but differs in how the boundary conditions are specified.

    Args:
        velocity: Vector field sampled on a grid
        obstacles: List of Obstacles to specify boundary conditions inside the domain (Default value = ())
        solve: Parameters for the pressure solve as.
        active: (Optional) Mask for which cells the pressure should be solved.
            If given, the velocity may take `NaN` values where it does not contribute to the pressure.
            Also, the total divergence will never be subtracted if active is given, even if all values are 1.

    Returns:
        velocity: divergence-free velocity of type `type(velocity)`
        pressure: solved pressure field, `CenteredGrid`
    """
    assert isinstance(obstacles, (tuple, list)), f"obstacles must be a tuple or list but got {type(obstacles)}"
    # Turn geometries into Obstacles
    obstacles = [Obstacle(o) if isinstance(o, Geometry) else o for o in obstacles]
    for obstacle in obstacles:
        assert obstacle.geometry.vector.item_names == velocity.vector.item_names, f"Obstacles must live in the same physical space as the velocity field {velocity.vector.item_names} but got {type(obstacle.geometry).__name__} obstacle with order {obstacle.geometry.vector.item_names}"
    input_velocity = velocity
    # --- Create masks ---
    accessible_extrapolation = _accessible_extrapolation(input_velocity.extrapolation)
    with NUMPY:
        accessible = CenteredGrid(~union([obs.geometry for obs in obstacles]), accessible_extrapolation,
                                  velocity.bounds, velocity.resolution)
        hard_bcs = field.stagger(accessible, math.minimum, input_velocity.extrapolation, type=type(velocity))
    all_active = active is None
    if active is None:
        active = accessible.with_extrapolation(extrapolation.NONE)
    else:
        active *= accessible  # no pressure inside obstacles
    # --- Linear solve ---
    velocity = apply_boundary_conditions(velocity, obstacles)
    div = divergence(velocity) * active
    if not all_active:  # NaN in velocity allowed
        div = field.where(field.is_finite(div), div, 0)
    if not input_velocity.extrapolation.is_flexible and all_active:
        assert solve.preprocess_y is None, "fluid.make_incompressible() does not support custom preprocessing"
        solve = copy_with(solve, preprocess_y=_balance_divergence, preprocess_y_args=(active,))
    if solve.x0 is None:
        pressure_extrapolation = _pressure_extrapolation(input_velocity.extrapolation)
        solve = copy_with(solve, x0=CenteredGrid(0, pressure_extrapolation, div.bounds, div.resolution))
    if batch(math.merge_shapes(*obstacles)).without(
            batch(solve.x0)):  # The initial pressure guess must contain all batch dimensions
        solve = copy_with(solve, x0=expand(solve.x0, batch(math.merge_shapes(*obstacles))))
    pressure = math.solve_linear(masked_laplace, f_args=[hard_bcs, active], y=div, solve=solve)
    # --- Subtract grad p ---
    grad_pressure = field.spatial_gradient(pressure, input_velocity.extrapolation, type=type(velocity)) * hard_bcs
    velocity = (velocity - grad_pressure).with_extrapolation(input_velocity.extrapolation)
    return velocity, pressure


def make_incompressible_two_way(velocity: GridType,
                                obstacles: tuple[Obstacle] | list[Obstacle] = (),
                                solve=math.Solve('auto', 1e-5, 1e-5, gradient_solve=math.Solve('auto', 1e-5, 1e-5)),
                                active: CenteredGrid = None) -> Tuple[GridType, CenteredGrid]:
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.

    This method is similar to :func:`field.divergence_free()` but differs in how the boundary conditions are specified.

    Args:
        velocity: Vector field sampled on a grid
        obstacles: List of Obstacles to specify boundary conditions inside the domain (Default value = ())
        solve: Parameters for the pressure solve as.
        active: (Optional) Mask for which cells the pressure should be solved.
            If given, the velocity may take `NaN` values where it does not contribute to the pressure.
            Also, the total divergence will never be subtracted if active is given, even if all values are 1.

    Returns:
        velocity: divergence-free velocity of type `type(velocity)`
        pressure: solved pressure field, `CenteredGrid`
    """
    assert isinstance(obstacles, (tuple, list)), f"obstacles must be a tuple or list but got {type(obstacles)}"
    obstacles = [Obstacle(o) if isinstance(o, Geometry) else o for o in obstacles]
    for obstacle in obstacles:
        assert obstacle.geometry.vector.item_names == velocity.vector.item_names, f"Obstacles must live in the same physical space as the velocity field {velocity.vector.item_names} but got {type(obstacle.geometry).__name__} obstacle with order {obstacle.geometry.vector.item_names}"
    input_velocity = velocity
    # --- Create masks ---
    accessible_extrapolation = _accessible_extrapolation(input_velocity.extrapolation)
    with NUMPY:
        accessible = CenteredGrid(~union([obs.geometry for obs in obstacles]), accessible_extrapolation,
                                  velocity.bounds, velocity.resolution)
        hard_bcs = field.stagger(accessible, math.minimum, input_velocity.extrapolation, type=type(velocity))
    all_active = active is None
    if active is None:
        active = accessible.with_extrapolation(extrapolation.NONE)
    else:
        active *= accessible  # no pressure inside obstacles
    # --- Linear solve ---
    velocity, obstacles = apply_boundary_conditions_two_way(velocity, density=None, obstacles=obstacles)
    div = divergence(velocity) * active
    if not all_active:  # NaN in velocity allowed
        div = field.where(field.is_finite(div), div, 0)
    if not input_velocity.extrapolation.is_flexible and all_active:
        assert solve.preprocess_y is None, "fluid.make_incompressible() does not support custom preprocessing"
        solve = copy_with(solve, preprocess_y=_balance_divergence, preprocess_y_args=(active,))
    if solve.x0 is None:
        pressure_extrapolation = _pressure_extrapolation(input_velocity.extrapolation)
        solve = copy_with(solve, x0=CenteredGrid(0, pressure_extrapolation, div.bounds, div.resolution))
    if batch(math.merge_shapes(*obstacles)).without(
            batch(solve.x0)):  # The initial pressure guess must contain all batch dimensions
        solve = copy_with(solve, x0=expand(solve.x0, batch(math.merge_shapes(*obstacles))))
    pressure = math.solve_linear(masked_laplace, f_args=[hard_bcs, active], y=div, solve=solve)
    # --- Subtract grad p ---
    grad_pressure = field.spatial_gradient(pressure, input_velocity.extrapolation, type=type(velocity)) * hard_bcs
    velocity = (velocity - grad_pressure).with_extrapolation(input_velocity.extrapolation)
    return velocity, pressure


@math.jit_compile_linear  # jit compilation is required for boundary conditions that add a constant offset solving Ax + b = y
def masked_laplace(pressure: CenteredGrid, hard_bcs: Grid, active: CenteredGrid) -> CenteredGrid:
    """
    Computes the laplace of `pressure` in the presence of obstacles.

    Args:
        pressure: Pressure field.
        hard_bcs: Mask encoding which cells are connected to each other.
            One between fluid cells, zero inside and at the boundary of obstacles.
            This should be of the same type as the velocity, i.e. `StaggeredGrid` or `CenteredGrid`.
        active: Mask indicating for which cells the pressure value is valid.
            Linear solves will only determine the pressure for these cells.
            This is generally zero inside obstacles and in non-simulated regions.

    Returns:
        `CenteredGrid`
    """
    grad = spatial_gradient(pressure, extrapolation.NONE, type=type(hard_bcs))
    valid_grad = grad * field.bake_extrapolation(hard_bcs)
    div = divergence(valid_grad)
    laplace = where(active, div, pressure)
    return laplace


def _balance_divergence(div, active):
    return div - active * (field.mean(div) / field.mean(active))


def apply_boundary_conditions(velocity: Grid | PointCloud, obstacles: tuple | list) -> Grid | PointCloud:
    for obstacle in obstacles:
        if isinstance(obstacle, Geometry):
            obstacle = Obstacle(obstacle)
        assert isinstance(obstacle, Obstacle)
        # Samples the overlap ratio of the obstacle with sample points on the velocity grid
        obs_mask = SoftGeometryMask(obstacle.geometry, balance=1) @ velocity
        velocity_field_before = velocity
        if obstacle.is_stationary:
            # Stationary obstacles are treated as hard boundaries, so we set the velocity to zero where the obstacle is
            velocity = (1 - obs_mask) * velocity
        else:
            # Construct an angular velocity field (pure curl) centered in the center of the obstacle
            angular_velocity = AngularVelocity(location=obstacle.geometry.center, strength=obstacle.angular_velocity,
                                               falloff=None) @ velocity
            # Velocities outside the object are untouched, but the velocities inside the object are set to the angular velocity + the linear velocity of the object
            # The linear velocity is constant over the entire object, but the angular velocity depends on how far away from the center of the object we are
            velocity = (1 - obs_mask) * velocity + obs_mask * (angular_velocity + obstacle.velocity)
    return velocity


def apply_boundary_conditions_two_way(velocity: Grid | PointCloud, density: Grid | PointCloud, obstacles: tuple | list):
    """
    NOTE: This method only works in 2D for the time being.
    The curl equation is only valid in 2D.

    Enforces velocities boundary conditions on a velocity grid.
    Cells inside obstacles will get their velocity from the obstacle movement.
    Cells outside far away will be unaffected.

    Args:
      velocity: Velocity `Grid`.
      density: Density `Grid`.
      obstacles: Obstacles as `tuple` or `list`

    Returns:
        Velocity of same type as `velocity`
    """
    new_obstacles = []
    for obstacle in obstacles:
        assert isinstance(obstacle, Obstacle), 'Two way boundary conditions only work with Obstacle objects'
        # Samples the overlap ratio of the obstacle with sample points on the velocity grid
        obs_mask = SoftGeometryMask(obstacle.geometry, balance=1) @ velocity
        velocity_field_before = velocity
        if obstacle.is_stationary:
            # Stationary obstacles are treated as hard boundaries, so we set the velocity to zero where the obstacle is
            velocity = (1 - obs_mask) * velocity
        else:
            # Construct an angular velocity field (pure curl) centered in the center of the obstacle
            angular_velocity = AngularVelocity(location=obstacle.geometry.center, strength=obstacle.angular_velocity,
                                               falloff=None) @ velocity
            # Velocities outside the object are untouched, but the velocities inside the object are set to the angular velocity + the linear velocity of the object
            # The linear velocity is constant over the entire object, but the angular velocity depends on how far away from the center of the object we are
            velocity = (1 - obs_mask) * velocity + obs_mask * (angular_velocity + obstacle.velocity)

        change_in_velocity = velocity - velocity_field_before
        # # Now we need to set the velocity and the angular velocity of the obstacle based on the change in momentum of the fluid
        # # The scalar linear velocity of the obstacle should change by the average change in velocity of the fluid inside the obstacle
        masked_velocity = change_in_velocity * obs_mask  # Note that this allows fractional values on the boundary
        new_velocity = obstacle.velocity - math.sum(masked_velocity.data) / obstacle.geometry.volume

        # # The angular velocity of the obstacle should change by the average change in angular velocity of the fluid inside the obstacle
        distance_vector_from_center = velocity.points - obstacle.geometry.center
        # Compute the cross product of the distance vector and the change in velocity
        cross_product = math.cross_product(distance_vector_from_center, masked_velocity.data)

        # Sum all the z-components of the cross products
        new_angular_velocity = obstacle.angular_velocity - math.sum(cross_product[..., 2:3]) / obstacle.geometry.volume

        new_obstacle = obstacle.copied_with(velocity=new_velocity, angular_velocity=new_angular_velocity)
        new_obstacles.append(new_obstacle)

    return velocity, new_obstacles


def boundary_push(particles: PointCloud, obstacles: tuple | list, offset: float = 0.5) -> PointCloud:
    """
    Enforces boundary conditions by correcting possible errors of the advection step and shifting particles out of
    obstacles or back into the domain.

    Args:
        particles: PointCloud holding particle positions as elements
        obstacles: List of `Obstacle` or `Geometry` objects where any particles inside should get shifted outwards
        offset: Minimum distance between particles and domain boundary / obstacle surface after particles have been shifted.

    Returns:
        PointCloud where all particles are inside the domain / outside of obstacles.
    """
    pos = particles.elements.center
    for obj in obstacles:
        geometry = obj.geometry if isinstance(obj, Obstacle) else obj
        assert isinstance(geometry,
                          Geometry), f"obstacles must be a list of Obstacle or Geometry objects but got {type(obj)}"
        pos = geometry.push(pos, shift_amount=offset)
    return particles.with_elements(particles.elements @ pos)


def _pressure_extrapolation(vext: Extrapolation):
    if vext == extrapolation.PERIODIC:
        return extrapolation.PERIODIC
    elif vext == extrapolation.BOUNDARY:
        return extrapolation.ZERO
    elif isinstance(vext, extrapolation.ConstantExtrapolation):
        return extrapolation.BOUNDARY
    else:
        return extrapolation.map(_pressure_extrapolation, vext)


def _accessible_extrapolation(vext: Extrapolation):
    """ Determine whether outside cells are accessible based on the velocity extrapolation. """
    if vext == extrapolation.PERIODIC:
        return extrapolation.PERIODIC
    elif vext == extrapolation.BOUNDARY:
        return extrapolation.ONE
    elif isinstance(vext, extrapolation.ConstantExtrapolation):
        return extrapolation.ZERO
    elif isinstance(vext, FieldEmbedding):
        return extrapolation.ONE
    elif isinstance(vext, extrapolation._MixedExtrapolation):
        return combine_sides(**{dim: (_accessible_extrapolation(lo), _accessible_extrapolation(hi)) for dim, (lo, hi) in
                                vext.ext.items()})
    elif isinstance(vext, extrapolation._NormalTangentialExtrapolation):
        return _accessible_extrapolation(vext.normal)
    else:
        raise ValueError(f"Unsupported extrapolation: {type(vext)}")
