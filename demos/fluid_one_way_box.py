"""
Simple example for two-way solve with a box suspended in fluids.
"""
from phi.flow import *
from phi.torch.flow import *
# from phi.jax.flow import *

# Define a domain with 128x128 cells and 100x100 units
DOMAIN = dict(x=128, y=128, bounds=Box(x=100, y=100))

# We place a single box in the middle of the domain, free to be pushed around by the incompressible fluid
# The geometry is defined as a single box
BOX_GEOMETRY = Box(x=(40, 60), y=(40, 60))

OBSTACLES = [Obstacle(BOX_GEOMETRY, moment_of_inertia=1e9, mass=1e2)]
# OBSTACLE_MASK = HardGeometryMask(OBSTACLE.geometry) @ CenteredGrid(0, extrapolation.BOUNDARY, **DOMAIN)

INFLOW = (
        # CenteredGrid(Box['x,y', 14:21, 6:10], extrapolation.BOUNDARY, **DOMAIN)
        # + CenteredGrid(Box['x,y', 81:88, 6:10], extrapolation.BOUNDARY, **DOMAIN) * 0.9
        CenteredGrid(Box['x,y', 44:47, 6:10], extrapolation.BOUNDARY, **DOMAIN) * 2.0
)
velocity = StaggeredGrid(0, extrapolation.ZERO, **DOMAIN)
smoke = pressure = divergence = remaining_divergence = CenteredGrid(0, extrapolation.BOUNDARY, **DOMAIN)


# @jit_compile
def step(smoke, velocity, pressure, obstacles, dt=1.):
    smoke = advect.semi_lagrangian(smoke, velocity, 1) + INFLOW
    buoyancy_force = smoke * (0, 0.1) @ velocity  # resamples density to velocity sample points
    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
    velocity, pressure, obstacles = fluid.make_incompressible_two_way(velocity, obstacles,
                                                                      Solve('CG-adaptive', 1e-3, 1e-3, x0=pressure))
    remaining_divergence = field.divergence(velocity)
    return smoke, velocity, pressure, obstacles, remaining_divergence


obstacles = OBSTACLES
OBSTACLE_MASK = CenteredGrid(obstacles[0].geometry, extrapolation.ZERO, **DOMAIN)
for _ in view('smoke, velocity, pressure, OBSTACLE_MASK', play=True, namespace=globals()).range(warmup=1):
    _, _, _, obstacles, _ = step(smoke, velocity, pressure, obstacles)
    smoke, velocity, pressure, _, remaining_divergence = step(smoke, velocity, pressure, [])
    OBSTACLE_MASK = CenteredGrid(obstacles[0].geometry, extrapolation.ZERO, **DOMAIN)
