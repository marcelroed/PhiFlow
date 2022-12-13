""" Rotating Bar
This demo shows how to simulate fluid flow with moving or rotating obstacles.
"""
from phi.flow import *
from phi.jax import *


DOMAIN = dict(x=100, y=100, bounds=Box(x=100, y=100))
DT = 1.0
obstacle = Obstacle(Box(x=(47, 53), y=(20, 70)), angular_velocity=0.05)
OBSTACLE_MASK = CenteredGrid(obstacle.geometry, 0, **DOMAIN)  # to show in user interface
velocity = StaggeredGrid(0, extrapolation.BOUNDARY, **DOMAIN)


@jit_compile
def step(obstacle, velocity):
    obstacle = obstacle.copied_with(geometry=obstacle.geometry.rotated(-obstacle.angular_velocity * DT))  # rotate bar
    velocity = advect.mac_cormack(velocity, velocity, DT)
    print(math.max(field.curl(velocity).data, 'x,y'))
    velocity, pressure = fluid.make_incompressible(velocity, (obstacle,), Solve('CG-adaptive', 1e-5, 1e-5))
    fluid.masked_laplace.tracers.clear()  # we will need to retrace because the matrix changes each step. This is not needed when JIT-compiling the physics.
    OBSTACLE_MASK = CenteredGrid(obstacle.geometry, extrapolation.ZERO, **DOMAIN)
    return obstacle, velocity, OBSTACLE_MASK


for frame in view(velocity, OBSTACLE_MASK, namespace=globals(), framerate=10, display=('velocity', 'OBSTACLE_MASK')).range():
    obstacle, velocity, OBSTACLE_MASK = step(obstacle, velocity)
