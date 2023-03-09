"""
This tests a simple obstacle with force calculation and obstacle update.
We try compiling with JAX, since this has resulted in errors before.
"""

# Make sure we check for tracer leaks
import os
os.environ['JAX_CHECK_TRACER_LEAKS'] = '1'

from phi.flow import *
from phi.jax.flow import *

DOMAIN = dict(x=100, y=100, bounds=Box(x=100, y=100))

DT = 1.0

obstacles = [
    Obstacle(Box(x=(40, 60), y=(40, 60)), angular_velocity=0.05, mass=1e3)
]


@jit_compile
def step(obstacles, pressure):
    obstacle_forces = fluid.pressure_to_obstacles(velocity=None, pressure=pressure, obstacles=obstacles, dt=DT)
    obstacles = update_obstacles_forces(obstacles, obstacle_forces, dt=DT)
    return obstacles


pressure = CenteredGrid(0, extrapolation.ZERO, **DOMAIN)
obstacle_geometry = obstacles[0].geometry
i = 0
for s in view('pressure', namespace=globals()).range(warmup=1):
    print(i := i + 1)
    obstacles = step(obstacles, pressure=pressure)
    obstacle_geometry = obstacles[0].geometry
    print(s)
