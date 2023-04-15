"""
Functions for simulating incompressible fluids, both grid-based and particle-based.

The main function for incompressible fluids (Eulerian as well as FLIP / PIC) is `make_incompressible()` which removes the divergence of a velocity field.
"""
from numbers import Number
from typing import Tuple, Union, List

import torch

import phi.geom._transform
from phi import math, field
from phi.field import SoftGeometryMask, AngularVelocity, Grid, divergence, spatial_gradient, where, CenteredGrid, \
    PointCloud
from phi.geom import union, Geometry, Sphere, Box, subdivide_line_segment, LevelSet
from phi.math import wrap, channel, Tensor
from ..field._embed import FieldEmbedding
from ..field._grid import GridType
from ..math import extrapolation, NUMPY, batch, shape, non_channel, expand, spatial
from ..math._magic_ops import copy_with
from ..math.extrapolation import combine_sides, Extrapolation
from pytorch3d.ops.marching_cubes import marching_cubes


class ForceSchedule:
    pass


class ObstacleUpdate:
    def __init__(self, delta_net_momentum: Tensor, delta_angular_momentum: float):
        self.delta_net_momentum = delta_net_momentum
        self.delta_angular_momentum = delta_angular_momentum

    def __repr__(self):
        return f'ObstacleUpdate({self.delta_net_momentum}, {self.delta_angular_momentum})'


class ObstacleForce:
    def __init__(self,
                 force: Tensor,
                 torque: Tensor,  # Scalar tensor
                 ):
        self.force = force
        self.torque = torque

    def __str__(self):
        return f'ObstacleForce(force={self.force}, torque={self.torque})'

    def __repr__(self):
        return f'ObstacleForce(force={self.force}, torque={self.torque})'


class Obstacle:
    """
    An obstacle defines boundary conditions inside a geometry.
    It can also have a linear and angular velocity.
    """

    def __repr__(self):
        return f'Obstacle(geometry={self.geometry}, velocity={self.velocity}, angular_velocity={self.angular_velocity})'

    def __str__(self):
        return f'Obstacle(geometry={self.geometry}, velocity={self.velocity}, angular_velocity={self.angular_velocity})'

    def __init__(self, geometry: Geometry, velocity: Union[Tensor, Tuple, List, None, Number] = None,
                 angular_velocity: float = 0.0, mass: float = 1.0,
                 moment_of_inertia: float = 1.0):
        """
        Args:
            geometry: Physical shape and size of the obstacle.
            velocity: Linear velocity vector of the obstacle.
            angular_velocity: Rotation speed of the obstacle. Scalar value in 2D, vector in 3D.
        """
        self.geometry: Geometry = geometry
        if velocity is None:
            velocity = 0
        if isinstance(velocity, Number):
            velocity = [float(velocity)] * channel(geometry).size

        self.velocity = wrap(velocity, channel(geometry)) if isinstance(velocity, (tuple, list)) else velocity
        self.angular_velocity = angular_velocity
        self.shape = shape(geometry) & non_channel(self.velocity) & non_channel(angular_velocity)
        # self.center_mass = center_mass if not center_mass is None else self.geometry.center
        # self.geometry.set_center(self.center_mass)
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia

    def __eq__(self, other: 'Obstacle'):
        return all([getattr(self, attr) == getattr(other, attr) for attr in self.__variable_attrs__()])

    def __variable_attrs__(self):
        return 'geometry', 'velocity', 'angular_velocity', 'shape', 'mass', 'moment_of_inertia'

    @property
    def is_stationary(self):
        """ Test whether the obstacle is completely still. """
        return isinstance(self.velocity, (int, float)) and self.velocity == 0 and isinstance(self.angular_velocity, (
            int, float)) and self.angular_velocity == 0

    def copied_with(self, **kwargs):
        geometry, velocity, angular_velocity, mass, moment_of_inertia = self.geometry, self.velocity, self.angular_velocity, self.mass, self.moment_of_inertia
        if 'geometry' in kwargs:
            geometry = kwargs['geometry']
        if 'velocity' in kwargs:
            velocity = kwargs['velocity']
        if 'angular_velocity' in kwargs:
            angular_velocity = kwargs['angular_velocity']
        return Obstacle(geometry=geometry, velocity=velocity, angular_velocity=angular_velocity, mass=mass,
                        moment_of_inertia=moment_of_inertia)

    def update_copy(self, obstacle_update: ObstacleUpdate, dt: float = 1.):
        new_velocity = self.velocity + obstacle_update.delta_net_momentum / dt / self.mass  # F = dp / dt
        new_angular_velocity = self.angular_velocity + obstacle_update.delta_angular_momentum / self.moment_of_inertia
        # if (new_velocity ** 2).sum.sqrt() > 0.1:
        #     new_velocity = new_velocity * 0.1 / (new_velocity ** 2).sum.sqrt()
        # if (abs(new_angular_velocity) > 0.1).all:
        #     new_angular_velocity = new_angular_velocity * 0.1 / abs(new_angular_velocity)
        new_geometry = self.geometry.shifted(dt * new_velocity).rotated(dt * new_angular_velocity)
        print('Delta_net_momentum', obstacle_update.delta_net_momentum)
        print('Moved by', new_geometry.center - self.geometry.center)
        # print(f'Moving the geometry by {new_velocity * dt} and rotating by {dt * new_angular_velocity}')
        return self.copied_with(geometry=new_geometry, velocity=new_velocity, angular_velocity=0)

    def update_copy_forces(self, obstacle_force: ObstacleForce, dt: float = 1.0):
        new_velocity = self.velocity + obstacle_force.force / dt / self.mass
        new_angular_velocity = self.angular_velocity + obstacle_force.torque / self.moment_of_inertia
        new_geometry = self.geometry.shifted(dt * new_velocity).rotated(dt * new_angular_velocity)
        print(f'Moving the geometry by {new_velocity * dt} and rotating by {dt * new_angular_velocity}')
        return self.copied_with(geometry=new_geometry, velocity=new_velocity, angular_velocity=new_angular_velocity)

    def update_copy_set(self, obstacle_update: ObstacleUpdate, dt: float = 1.):
        new_velocity = obstacle_update.delta_net_momentum / self.mass
        new_angular_velocity = obstacle_update.delta_angular_momentum / self.moment_of_inertia

        new_geometry = self.geometry.shifted(dt * new_velocity).rotated(dt * new_angular_velocity)
        print('Moved by', new_geometry.center - self.geometry.center)
        # print(f'Moving the geometry by {new_velocity * dt} and rotating by {dt * new_angular_velocity}')
        return self.copied_with(geometry=new_geometry, velocity=new_velocity, angular_velocity=new_angular_velocity)


def update_obstacles(obstacles: List[Obstacle], obstacle_updates: List[ObstacleUpdate], dt: float = 1.):
    print(obstacle_updates)
    return [
        obstacle.update_copy(obstacle_update, dt)
        for obstacle, obstacle_update in zip(obstacles, obstacle_updates)
    ]


def update_obstacles_forces(obstacles: List[Obstacle], obstacle_forces: List[ObstacleForce], dt: float = 1.):
    print(obstacle_forces)
    return [
        obstacle.update_copy_forces(obstacle_force, dt)
        for obstacle, obstacle_force in zip(obstacles, obstacle_forces)
    ]


def sample_at_edges(sample_field, edges):
    # TODO(marcelroed): Should be this simple, so can remove function definition
    return sample_field._sample(geometry=edges, scheme=field.numerical.Scheme())


def pressure_integral(pressure, edges):
    # TODO(marcelroed): Generalize this to 3D

    # Currently implements line sampling for the edges of a 2D polygon
    sampled_pressure = sample_at_edges(pressure, edges)
    pressure_integrals = sampled_pressure * edges.length()
    return pressure_integrals


def delta_tilde_func(values, eps=0.5):
    # TODO (marcelroed): What happens when the object isn't convex? In this case the distance values will drop off
    #  slower than they should, and the integral will be greater than we want. Does the normalization of delta deal
    #  with this case? Ignore for now.
    """
    A smeared out one-sided delta function. This function is the derivative of the smoothed heaviside function.
    H_eps(x) = 1 / (1 + exp(-x / eps))
    => delta_eps(x) = H_eps'(x) = exp(x / eps) / (eps * (1 + exp(x / eps)) ** 2)
    """
    delta = math.exp(values / eps) / (eps * (1 + math.exp(values / eps)) ** 2)
    delta = math.where((values < 10) & (values > -10), delta, 0)
    return 2 * math.where(values > 0, delta, 0)


def delta_hat_func(phi_val, eps=5.0):
    """This is the derivative of a one-sided smeared out heaviside function. Since we want to only measure the
    pressure on the positive side of the level set, we need to double the value on that side for the integral to be
    correct.
    """
    return math.where(
        (0 <= phi_val) & (phi_val <= eps),
        1 / eps + math.cos(math.pi * phi_val / eps) / eps,
        0)


def compute_area_normals(verts: torch.FloatTensor, faces: torch.LongTensor):
    # Verts: (N, 3), faces: (M, 3)


    # TODO(marcelroed): Determine dim, since this defaults to the first 3-dimensional axis
    face_normals = 1 / 2 * torch.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    # face_normals: (M, 3), area-normals for each face

    return face_normals


def get_area_normals(grid: CenteredGrid, level_set: LevelSet) -> CenteredGrid:
    """Gets the equivalent normal scaled by the area of the region inside each cell"""
    corner_positions = grid.element_corners().center
    level_set_values_corner = level_set.function(corner_positions)
    # Add batch dimension TODO(marcelroed): Make this use phiflow batch (and 1 when there's none)
    level_set_values_corner_torch = level_set_values_corner.native(level_set_values_corner.shape)[None, ...]
    # TODO(marcelroed): marching_cubes is broken on CPU, so need to fix it or always run on GPU. Both versions
    #  currently return floats for vert indices when not in local coords, so need to fix that too.
    #  Finally, this should return the cell indices for each face, so that we can reduce the area normals.
    verts, faces = marching_cubes(level_set_values_corner_torch, isolevel=0.0, return_local_coords=False)
    verts, faces = verts[0].long(), faces[0]
    # TODO(marcelroed): Improve the verts by moving them to the zero of the level set, imbue them with gradients

    area_normals = compute_area_normals(verts, faces)

    # TODO(marcelroed): Reduce to the right cells
    cell_normals = torch.zeros(*[s - 1 for s in corner_positions.shape], dtype=corner_positions.dtype, device=corner_positions.device)

    cell_normals.scatter_add_(dim=0, index=face_belongs_to_cell, src=area_normals)

    return grid.with_values(cell_normals)

    # return math.tensor(cell_normals, *grid.elements.center.shape)



def level_set_pressure_integral(pressure_field: CenteredGrid, level_set: LevelSet, centroid):
    # pos = pressure_field.elements.center
    # level_set_values = level_set.function(pos)
    # level_set_values, level_set_gradients = math.gradient(level_set.function, wrt='x', get_output=True)(corner_positions)


    # Normal vectors for each cell, length is by their projected area
    # TODO(marcelroed): Should certainly be sparse!
    area_normals = get_area_normals(grid=pressure_field, level_set=level_set)

    # dV = pressure_field.elements.volume
    pressure_field._masked_sample(area_normals.elements.center, mask=)
    pressure_integrand = pressure_field.masked_sample() * area_normals
    linear_force = - math.sum(pressure_integrand, dim=spatial(pressure_integrand))
    torque = - math.sum(math.cross_product(pressure_integrand, pressure_integrand.elements.center - centroid), dim=spatial(pressure_integrand))
    return linear_force, torque


def pressure_to_obstacles(velocity, pressure: CenteredGrid, obstacles: List[Obstacle], dt: float) -> List[
    ObstacleForce]:
    obstacle_forces = []
    for obstacle in obstacles:  # TODO(marcelroed): vectorize by using geometry stacks? Might not be possible depending on uniformity.
        geometry = obstacle.geometry
        if isinstance(geometry, (Box, Sphere, phi.geom._transform.RotatedGeometry)):
            # Construct a field of distances to the center of mass for torque calculations
            distance_vec_to_centroid = field.CenteredGrid(pressure.elements.center - geometry.center_of_mass)

            # linear_force = 0
            # torque = 0

            edges = geometry.get_edges()
            # Subdivide edges to get a more accurate pressure integral
            edges = subdivide_line_segment(edges, 20)

            normals = geometry.get_normals()

            # Calculate integral of pressure over the edge
            linear_forces = - pressure_integral(pressure, edges) * normals
            total_linear_force = math.sum(linear_forces, ('b', 'edges'))
            # Calculate torque
            torques = math.cross_product(distance_vec_to_centroid._sample(edges, scheme=field.numerical.Scheme()),
                                         linear_forces)
            total_torque = math.sum(torques, ('b', 'edges', 'vector'))  # Should be zero in all other components

            obstacle_forces.append(ObstacleForce(force=total_linear_force, torque=total_torque))
        elif isinstance(geometry, LevelSet):
            # Calculate integral of pressure over the boundary of the level set
            linear_force, torque = level_set_pressure_integral(pressure, geometry, centroid=geometry.center_of_mass)
            obstacle_forces.append(ObstacleForce(force=linear_force, torque=torque))
        else:
            raise NotImplementedError(f'Obstacle type {type(geometry)} not implemented yet')

    # Then apply the obstacle forces
    return obstacle_forces


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
                                obstacles: Union[Tuple[Obstacle], List[Obstacle]] = (),
                                solve=math.Solve('auto', 1e-5, 1e-5, gradient_solve=math.Solve('auto', 1e-5, 1e-5)),
                                active: CenteredGrid = None,
                                fluid_density: float = 1.0,
                                dt: float = 1.0) -> Tuple[GridType, CenteredGrid, List[Obstacle]]:
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
        fluid_density: A single number indicating the density of the incompressible fluid. This will affect
            how much the fluid pushes obstacles around.
        dt: The time step to use for the obstacle updates.

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
    velocity, obstacle_updates = apply_boundary_conditions_two_way(velocity, density=fluid_density, obstacles=obstacles)
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

    # We solve the linear system Δp = div(v) * active for p
    pressure = math.solve_linear(masked_laplace, f_args=[hard_bcs, active], y=div, solve=solve)

    # Subtract the gradient of the pressure from the velocity to get the updated velocity
    grad_pressure = field.spatial_gradient(pressure, input_velocity.extrapolation, type=type(velocity)) * hard_bcs
    velocity = (velocity - grad_pressure).with_extrapolation(input_velocity.extrapolation)

    # Update obstacles
    new_obstacles = update_obstacles(obstacles, obstacle_updates, dt)
    return velocity, pressure, new_obstacles


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
    # First compute the spatial gradient of the pressure
    grad = spatial_gradient(pressure, extrapolation.NONE, type=type(hard_bcs))

    # The gradient is only defined for fluid cells, so we need to multiply it with the hard_bcs mask
    # We bake_extrapolation to get an additional line of cells around the grid that ensure our boundary conditions are correct
    valid_grad = grad * field.bake_extrapolation(hard_bcs)

    # Compute the divergence of the gradient in the valid regions
    div = divergence(valid_grad)

    # Now the laplace of the pressure will be the divergence of the gradient in valid regions, in other places we set it equal to the pressure
    # TODO(marcelroed): Why set it to the pressure otherwise?
    laplace = where(active, div, pressure)
    return laplace


def _balance_divergence(div, active):
    return div - active * (field.mean(div) / field.mean(active))


def apply_boundary_conditions(velocity: Union[Grid, PointCloud], obstacles: Union[tuple, list]) -> Union[
    Grid, PointCloud]:
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


def apply_boundary_conditions_two_way(velocity: Union[Grid, PointCloud], density: Union[Grid, PointCloud, float],
                                      obstacles: Union[tuple, list]):
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
    obstacle_updates = []
    for obstacle in obstacles:
        assert isinstance(obstacle, Obstacle), 'Two way boundary conditions only work with Obstacle objects'
        # Samples the overlap ratio of the obstacle with sample points on the velocity grid
        print('Change_in_momentum', type(velocity))
        obs_mask = SoftGeometryMask(obstacle.geometry, balance=0.5) @ velocity
        velocity_field_before = velocity
        if obstacle.is_stationary:
            # Stationary obstacles are treated as hard boundaries, so we set the velocity to zero where the obstacle is
            velocity = (1 - obs_mask) * velocity
            print('Object is stationary')
        else:
            print('Object is moving')
            # Construct an angular velocity field (pure curl) centered in the center of the obstacle
            angular_velocity = AngularVelocity(location=obstacle.geometry.center, strength=obstacle.angular_velocity,
                                               falloff=None) @ velocity
            # Velocities outside the object are untouched, but the velocities inside the object are set to the angular velocity + the linear velocity of the object
            # The linear velocity is constant over the entire object, but the angular velocity depends on how far away from the center of the object we are
            velocity_absorbed = obs_mask * velocity  # FIXME: Should this be assigned before or after the velocity updates?
            velocity = (1 - obs_mask) * velocity + obs_mask * (angular_velocity + obstacle.velocity) / density

            # # Now we need to set the velocity and the angular velocity of the obstacle based on the change in momentum of the fluid
            # # The scalar linear velocity of the obstacle should change by the average change in velocity of the fluid inside the obstacle

            distance_vector_from_center = velocity_absorbed.at_centers().elements.center - obstacle.geometry.center_of_mass
            change_in_momentum = (math.mean(velocity_absorbed.data, spatial('x,y'))) * density
            # # The angular velocity of the obstacle should change by the average change in angular velocity of the fluid inside the obstacle
            # Compute the cross product of the distance vector and the change in velocity

            cross_product = math.cross_product(distance_vector_from_center, velocity_absorbed.at_centers().data)
            cross_product_before = math.cross_product(distance_vector_from_center,
                                                      velocity_field_before.at_centers().data)

            # # Sum all the z-components of the cross products
            change_in_angular_momentum = - math.sum(cross_product - cross_product_before) * density

            # Approach storing the momentum in the field
            # change_in_momentum = math.sum(velocity.data) * density
            # change_in_angular_momentum = math.sum(math.cross_product(distance_vector_from_center, velocity.at_centers().data)) * density

            # Doing it with curls
            # total_curl = math.sum(field.curl(masked_velocity).data)
            # print(total_curl)
            # change_in_angular_momentum = - total_curl

            obstacle_updates.append(
                ObstacleUpdate(delta_net_momentum=change_in_momentum, delta_angular_momentum=change_in_angular_momentum)
            )

    return velocity, obstacle_updates


def boundary_push(particles: PointCloud, obstacles: Union[Tuple[Obstacle, ...], List[Obstacle]],
                  offset: float = 0.5) -> PointCloud:
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
