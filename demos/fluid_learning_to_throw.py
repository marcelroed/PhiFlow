import os

import torch.jit

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['JAX_CHECK_TRACER_LEAKS'] = '1'
# os.environ['JAX_DISABLE_JIT'] = '1'

from typing import List
# from phi.flow import *
# from phi.jax.flow import *
from phi.torch.flow import *
from tqdm.auto import trange, tqdm
from warnings import filterwarnings

filterwarnings('error')
filterwarnings('default', category=DeprecationWarning)
filterwarnings('default', category=UserWarning)

filterwarnings('default', category=torch.jit.TracerWarning)  # This seems like it may be a problem


DOMAIN = dict(x=20, y=20, bounds=Box(x=100, y=100))
DT = 0.1
G = 0.5
obstacle = Obstacle(Box(x=(20, 40), y=(20, 40)), angular_velocity=0.000, mass=1e10, moment_of_inertia=1e100)
# obstacle = Obstacle(LevelSet(), angular_velocity=0.000, mass=1e10, moment_of_inertia=1e100)
initial_obstacles = [obstacle]
# OBSTACLE_MASK = CenteredGrid(obstacle.geometry, 0, **DOMAIN)  # to show in user interface
# velocity = StaggeredGrid(0,
#                          extrapolation=extrapolation.combine_sides(
#                              x=extrapolation.ConstantExtrapolation(math.tensor([1.0, 0.0], channel(vector='x,y'))),
#                              y=extrapolation.ZERO),
#                          **DOMAIN)

original_geometry = obstacle.geometry
pressure = None



# Try to keep the box in the center of the domain in the final timestep
def simulate(obstacle_shift, initial_velocity: StaggeredGrid):
    pressure = None
    obstacles = [math.copy_with(obstacle, geometry=obstacle.geometry.shifted(obstacle_shift)) for obstacle in initial_obstacles]
    def step_sim(obstacles, velocity, pressure):
        velocity = advect.semi_lagrangian(velocity, velocity, DT)
        velocity, pressure = fluid.make_incompressible(velocity, obstacles, Solve('CG-adaptive', 1e-5, 1e-5, x0=None))
        obstacle_forces = fluid.pressure_to_obstacles(velocity, pressure, obstacles, dt=DT)
        # Add gravity to the obstacle
        obstacle_forces = [obstacle_force.copy_with(force=obstacle_force.force + math.tensor([0, -0], channel(vector='x,y')))
                           for obstacle_force in obstacle_forces]
        # print(obstacle_forces)
        # obstacle_forces[0].force = obstacle_forces[0].force +
        obstacles = update_obstacles_forces(obstacles, obstacle_forces=obstacle_forces)
        # print(obstacles)
        return obstacles, velocity, pressure

    # this exists, but stops gradients... that could be changed
    # math.choose_backend(velocity).while_loop(lambda i, obstacles: (i + 1, step_sim(obstacles, velocity)))

    # obstacles = jax.lax.fori_loop(0, 20, lambda i, obstacles: step_sim(obstacles, velocity), obstacles)
    velocity = initial_velocity
    # obstacles = initial_obstacles
    for _ in trange(2):
        obstacles, velocity, pressure = step_sim(obstacles, velocity, pressure)
        print(obstacles)

    # Incentivize being close to the center of the domain at timestep 20
    loss = math.vec_squared(obstacles[0].geometry.center - math.tensor([50, 50], channel(vector='x,y')))
    return loss


# def dummy_simulation(initial_velocity):
#     return math.sum(math.vec_squared(initial_velocity.at_centers().values - math.tensor([50, 50], channel(vector='x,y')), channel(vector='x,y')))

# initial_vector_potential = CenteredGrid(0., **DOMAIN)  # Only has a z-component in 2D
# initial_velocity = field.curl(initial_vector_potential, type=StaggeredGrid)
# print(initial_velocity.values[0])
initial_velocity = StaggeredGrid(math.tensor([1., 0.], channel(vector='x,y')),
                                 extrapolation=extrapolation.ZERO_GRADIENT,
                                 **DOMAIN)


sim_grad = field.functional_gradient(simulate, wrt='obstacle_shift')
# dummy_sim_grad = field.functional_gradient(dummy_simulation, wrt='initial_velocity')

# 20 steps of optimization
print(initial_velocity.data)
# tensors = [tensor.native(tensor.shape) for tensor in initial_velocity.data._tensors]

@jit_compile
def run_optimization():
    position_shift = math.tensor([10., 10.], channel(vector='x,y'))
    best_shift = position_shift.native(position_shift.shape).detach()
    best_val = math.inf
    tensors = [position_shift.native(position_shift.shape)]
    optimizer = torch.optim.Adam(tensors, lr=1.0)
    for _ in trange(200, desc='Optimizing velocity field'):
        # Calculate the gradient of the loss w.r.t. the velocity
        val, grad = sim_grad(position_shift, initial_velocity)
        # val, grad = dummy_sim_grad(initial_velocity)
        # grad_tensors = [tensor.native(tensor.shape) for tensor in grad.data._tensors]
        print(val, grad)
        grad_tensors = [grad.native(grad.shape)]
        print(tensors, grad_tensors)
        # grad = sim_grad(obstacles, initial_velocity)
        for t, g in zip(tensors, grad_tensors):
            t.grad = g

        # Update the velocity
        # print(grad.at_centers().values)
        # initial_velocity = initial_velocity - grad
        print('Making optimizer step')
        optimizer.step()
        optimizer.zero_grad()
        # print(initial_velocity.at_centers().values)
        print(f'{val =}')
        if float(val) < best_val:
            best_val = float(val)
            best_shift = tensors[0].detach()
    return best_shift

print(run_optimization())

# Show the final result
# vis.plot(best_vel, size=(20, 10))
# vis.show()


# @jit_compile
def step(obstacles, velocity, frame):
    print(math.max(field.curl(velocity).data, 'x,y'))
    velocity = advect.mac_cormack(velocity, velocity, DT) + G * DT
    velocity, pressure = fluid.make_incompressible(velocity, obstacles)
    obstacle_forces = fluid.pressure_to_obstacles(velocity, pressure, obstacles, dt=DT)
    obstacles = update_obstacles_forces(obstacles, obstacle_forces=obstacle_forces)
    fluid.masked_laplace.tracers.clear()  # we will need to retrace because the matrix changes each step. This is not needed when JIT-compiling the physics.
    OBSTACLE_MASK = CenteredGrid(obstacles[0].geometry, extrapolation.ZERO, **DOMAIN)
    return obstacles, velocity, pressure, OBSTACLE_MASK

# for frame in view(velocity, OBSTACLE_MASK, namespace=globals(), framerate=10,
#                   display=('velocity', 'pressure', 'OBSTACLE_MASK')).range():
#     obstacles, velocity, pressure, OBSTACLE_MASK = step(obstacles, velocity, frame)
