""" Rotating Bar
This demo shows how to simulate fluid flow with moving or rotating obstacles.
"""
# from phi.torch.flow import *
from phi.jax.flow import *

DOMAIN = dict(x=100, y=100, bounds=Box(x=100, y=100))
DT = 1.0
obstacle = Obstacle(Box(x=(47, 53), y=(20, 70)), angular_velocity=0.05)
obstacle_mask = CenteredGrid(obstacle.geometry, 0, **DOMAIN)  # to show in user interface
velocity = StaggeredGrid(0, ZERO_GRADIENT, **DOMAIN)

@jit_compile
def step(obstacle, velocity):
    obstacle = math.copy_with(obstacle, geometry=obstacle.geometry.rotated(-obstacle.angular_velocity * DT))
    # obstacle = obstacle.copied_with(geometry=obstacle.geometry.rotated(-obstacle.angular_velocity * DT))  # rotate bar
    velocity = advect.mac_cormack(velocity, velocity, DT)
    velocity, pressure = fluid.make_incompressible(velocity, (obstacle,), Solve('CG-adaptive', 1e-5, 1e-5))
    obstacle_mask = CenteredGrid(obstacle.geometry, 0, **DOMAIN)
    return obstacle, velocity, pressure, obstacle_mask


for frame in view(velocity, obstacle_mask, namespace=globals(), framerate=10, display=('velocity', 'obstacle_mask'), gui='dash').range():
    obstacle, velocity, pressure, obstacle_mask = step(obstacle, velocity)
    # obstacle = math.copy_with(obstacle, geometry=obstacle.geometry.rotated(-obstacle.angular_velocity * DT))
    # velocity = advect.mac_cormack(velocity, velocity, DT)
    # velocity, pressure = fluid.make_incompressible(velocity, (obstacle,), Solve('CG-adaptive', 1e-5, 1e-5))
    # obstacle_mask = CenteredGrid(obstacle.geometry, 0, **DOMAIN)
