""" Fluid Logo
Incompressible fluid simulation with obstacles and buoyancy.
"""
from phi.flow import *
# from phi.torch.flow import *
# from phi.tf.flow import *
from phi.jax.flow import *

DOMAIN = dict(x=128, y=128, bounds=Box(x=100, y=100))

OBSTACLE_GEOMETRIES = [Box(x=(15 + x * 7, 15 + (x + 1) * 7), y=(41, 83)) for x in range(1, 10, 2)] + [Box['x,y', 43:50, 41:48], Box['x,y', 15:43, 83:90], Box['x,y', 50:85, 83:90]]
OBSTACLE = Obstacle(union(OBSTACLE_GEOMETRIES))
OBSTACLE_MASK = resample(OBSTACLE.geometry, to=CenteredGrid(0, ZERO_GRADIENT, **DOMAIN))

INFLOW = CenteredGrid(Box['x,y', 14:21, 6:10], ZERO_GRADIENT, **DOMAIN) + \
         CenteredGrid(Box['x,y', 81:88, 6:10], ZERO_GRADIENT, **DOMAIN) * 0.9 + \
         CenteredGrid(Box['x,y', 44:47, 49:51], ZERO_GRADIENT, **DOMAIN) * 0.4
velocity = StaggeredGrid(0, 0, **DOMAIN)
smoke = pressure = divergence = remaining_divergence = CenteredGrid(0, ZERO_GRADIENT, **DOMAIN)

@jit_compile
def step(smoke, velocity, pressure, dt=1.):
    smoke = advect.semi_lagrangian(smoke, velocity, 1) + INFLOW
    buoyancy_force = resample(smoke * (0, 0.1), to=velocity)
    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
    velocity, pressure = fluid.make_incompressible(velocity, (OBSTACLE,), Solve('CG-adaptive', 1e-5, x0=pressure))
    remaining_divergence = field.divergence(velocity)
    return smoke, velocity, pressure, remaining_divergence


for _ in view('smoke, velocity, pressure, OBSTACLE_MASK', play=True, namespace=globals()).range(warmup=1):
    smoke, velocity, pressure, remaining_divergence = step(smoke, velocity, pressure)
