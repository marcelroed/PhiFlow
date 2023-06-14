from typing import List
from phi.flow import *
from phi.torch.flow import *
from tqdm.auto import trange, tqdm

DOMAIN = dict(x=50, y=50, bounds=Box(x=100, y=100))
DT = 1.0
G = 0.5
obstacle = Obstacle(Box(x=(20, 40), y=(20, 40)), angular_velocity=0.05, mass=1e3, moment_of_inertia=1e10)
OBSTACLE_MASK = CenteredGrid(obstacle.geometry, 0, **DOMAIN)  # to show in user interface
# velocity = StaggeredGrid(0,
#                          extrapolation=extrapolation.combine_sides(
#                              x=extrapolation.ConstantExtrapolation(math.tensor([1.0, 0.0], channel(vector='x,y'))),
#                              y=extrapolation.ZERO),
#                          **DOMAIN)

original_geometry = obstacle.geometry
pressure = None

obstacles = [obstacle]


# Try to keep the box in the center of the domain in the final timestep
def simulate(obstacles: List[Obstacle], velocity: StaggeredGrid):
    for _ in trange(20):
        velocity = advect.semi_lagrangian(velocity, velocity, DT)
        velocity, pressure = fluid.make_incompressible(velocity, [])
        obstacle_forces = fluid.pressure_to_obstacles(velocity, pressure, obstacles, dt=DT)

        # Add gravity to the obstacle
        obstacle_forces[0].force = obstacle_forces[0].force + math.tensor([0, -G], channel(vector='x,y'))
        obstacles = update_obstacles_forces(obstacles, obstacle_forces=obstacle_forces)
        # fluid.masked_laplace.tracers.clear()  # we will need to retrace because the matrix changes each step. This is not needed when JIT-compiling the physics.

    # Incentivize being close to the center of the domain at timestep 20
    loss = math.vec_length(obstacles[0].geometry.center - math.tensor([50, 50], channel(vector='x,y'))) ** 2
    return loss


initial_velocity = StaggeredGrid(math.tensor([0., 1.], channel(vector='x,y')),
                                 extrapolation=extrapolation.ZERO,
                                 **DOMAIN)

sim_grad = field.functional_gradient(simulate, wrt='velocity')

# 20 steps of optimization
best_vel = initial_velocity
best_val = math.inf
for _ in trange(20, desc='Optimizing velocity field'):
    # Calculate the gradient of the loss w.r.t. the velocity
    val, grad = sim_grad(obstacles, initial_velocity)
    # grad = sim_grad(obstacles, initial_velocity)
    # Update the velocity
    initial_velocity = initial_velocity - grad * 0.2
    if float(val) < best_val:
        best_val = float(val)
        best_vel = initial_velocity

# Show the final result
vis.plot(best_vel, size=(20, 10))
vis.show()


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
