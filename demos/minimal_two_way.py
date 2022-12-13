from phi.flow import *
import jax.numpy as jnp
from phi.jax import *


DOMAIN = dict(x=100, y=100, bounds=Box(x=100, y=100))
DT = 1.0
obstacle = Obstacle(Box(x=(40, 60), y=(40, 60)), angular_velocity=0.05, mass=1e3, moment_of_inertia=1e10)
OBSTACLE_MASK = CenteredGrid(obstacle.geometry, 0, **DOMAIN)  # to show in user interface
velocity = StaggeredGrid(0,
                         extrapolation.ConstantExtrapolation(math.tensor([1.0, 0.0], channel(vector='x,y'))),
                         **DOMAIN)

original_geometry = obstacle.geometry
pressure = None

obstacles = [obstacle]


# @jit_compile
def step(obstacles, velocity, frame):
    print(math.max(field.curl(velocity).data, 'x,y'))
    velocity = advect.mac_cormack(velocity, velocity, DT)
    velocity, pressure = fluid.make_incompressible(velocity, obstacles, Solve('CG-adaptive', 1e-5, 1e-5))
    obstacle_forces = fluid.pressure_to_obstacles(velocity, pressure, obstacles, dt=DT)
    obstacles = update_obstacles_forces(obstacles, obstacle_forces=obstacle_forces)
    fluid.masked_laplace.tracers.clear()  # we will need to retrace because the matrix changes each step. This is not needed when JIT-compiling the physics.
    OBSTACLE_MASK = CenteredGrid(obstacles[0].geometry, extrapolation.ZERO, **DOMAIN)
    return obstacles, velocity, pressure, OBSTACLE_MASK


for frame in view(velocity, OBSTACLE_MASK, namespace=globals(), framerate=10, display=('velocity', 'pressure', 'OBSTACLE_MASK')).range():
    obstacles, velocity, pressure, OBSTACLE_MASK = step(obstacles, velocity, frame)
