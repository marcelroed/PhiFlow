from copy import copy
from typing import List
from phi.flow import *
from phi.torch.flow import *
from tqdm.auto import trange, tqdm

DOMAIN = dict(x=50, y=50, bounds=Box(x=100, y=100))
DT = 1.0
G = 0.5
LEARNING_RATE = 0.2
N_OPT_STEPS = 20
velocity = StaggeredGrid(math.tensor([0., -1.], channel(vector='x,y')),
                         extrapolation=extrapolation.ZERO,
                         **DOMAIN)


pressure = None


# Try to keep the box in the center of the domain in the final timestep
def simulate(initial_obstacle: Obstacle, velocity: StaggeredGrid):
    obstacles = [initial_obstacle]
    for _ in trange(20):
        velocity = advect.semi_lagrangian(velocity, velocity, DT)
        velocity, pressure = fluid.make_incompressible(velocity, [])
        obstacle_forces = fluid.pressure_to_obstacles(velocity, pressure, obstacles, dt=DT)

        # Add gravity to the obstacle
        obstacle_forces[0].force = obstacle_forces[0].force + math.tensor([0, -G], channel(vector='x,y'))
        obstacles = update_obstacles_forces(obstacles, obstacle_forces=obstacle_forces)
        fluid.masked_laplace.tracers.clear()  # we will need to retrace because the matrix changes each step. This is not needed when JIT-compiling the physics.

    # Incentivize being close to the center of the domain at timestep 20
    loss = math.vec_length(obstacles[0].geometry.center - math.tensor([50, 50], channel(vector='x,y'))) ** 2
    return loss


initial_obstacle = Obstacle(Box(x=(20, 40), y=(20, 40)), angular_velocity=0.05, mass=1e3, moment_of_inertia=1e10)

sim_grad = field.functional_gradient(simulate,
                                     wrt='initial_obstacle',
                                     get_output=True)  # Include the output of the simulation as well as the gradient

# 20 steps of optimization
best_obstacle = copy(initial_obstacle)
best_val = math.inf
for _ in trange(N_OPT_STEPS, desc='Optimizing the obstacle'):
    # Calculate the gradient of the loss w.r.t. the velocity
    print(f'Testing with obstacle location {initial_obstacle.geometry.center}')
    val, grad = sim_grad(initial_obstacle, velocity)
    if float(val) < best_val:
        best_val = float(val)
        best_obstacle = initial_obstacle
    # Update the obstacle parameters we want to change
    # Only change the x-position of the lower and upper corners of the box by the average of their gradients.
    mean_movement = (grad.geometry._lower + grad.geometry._upper) / 2
    print(f'Loss: {val}, Gradient: {mean_movement}')
    moved_lower = initial_obstacle.geometry._lower - LEARNING_RATE * mean_movement
    moved_upper = initial_obstacle.geometry._upper - LEARNING_RATE * mean_movement
    initial_obstacle.geometry._lower = math.where([True, False], moved_lower, initial_obstacle.geometry._lower)
    initial_obstacle.geometry._upper = math.where([True, False], moved_upper, initial_obstacle.geometry._upper)
    # grad = sim_grad(obstacles, initial_velocity)
    # initial_velocity = initial_velocity - grad * 0.2

# Show the final result
print(f'Best obstacle: {best_obstacle}')
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
