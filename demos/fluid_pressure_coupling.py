"""
Testing for interactions between fluid and obstacles using pressure.
"""
from phi.flow import *
from phi.physics.fluid import update_obstacles_forces
from phi.torch.flow import *
# from phi.jax.flow import *

# Define a domain with 128x128 cells and 100x100 units
DOMAIN = dict(x=64, y=64, bounds=Box(x=100, y=100))

# We place a single box in the middle of the domain, free to be pushed around by the incompressible fluid
# The geometry is defined as a single box
BOX_GEOMETRY = Box(x=(40, 60), y=(40, 60))

print(BOX_GEOMETRY.get_edges())
print(BOX_GEOMETRY.get_normals())

OBSTACLES = [Obstacle(BOX_GEOMETRY, moment_of_inertia=1e6, mass=1e2)]
# OBSTACLE_MASK = HardGeometryMask(OBSTACLE.geometry) @ CenteredGrid(0, extrapolation.BOUNDARY, **DOMAIN)

INFLOW = (
    # CenteredGrid(Box['x,y', 14:21, 6:10], extrapolation.BOUNDARY, **DOMAIN)
    # + CenteredGrid(Box['x,y', 81:88, 6:10], extrapolation.BOUNDARY, **DOMAIN) * 0.9
        CenteredGrid(Box['x,y', 44:47, 6:10], extrapolation.BOUNDARY, **DOMAIN) * 2.0
)
velocity = StaggeredGrid(0, extrapolation.BOUNDARY, **DOMAIN)
smoke = CenteredGrid(0, extrapolation.BOUNDARY, **DOMAIN)
pressure = None


# @jit_compile
def step(smoke, velocity, pressure, obstacles, dt):
    smoke = advect.mac_cormack(smoke, velocity, dt=dt) + INFLOW
    buoyancy_force = smoke * (0, 0.1) @ velocity  # resamples density to velocity sample points
    velocity = advect.mac_cormack(velocity, velocity, dt=dt) + buoyancy_force * dt
    print(math.max(field.curl(velocity).data, 'x,y'))
    velocity, pressure = fluid.make_incompressible(velocity, obstacles,
                                                                      Solve('CG-adaptive', 1e-5, 1e-5, x0=pressure))
    # Use the new pressure to compute the effects on the obstacles from the fluid
    obstacle_forces = fluid.pressure_to_obstacles(velocity, pressure, obstacles, dt=dt)
    obstacles = update_obstacles_forces(obstacles, obstacle_forces, dt=dt)
    # remaining_divergence = field.divergence(velocity)
    fluid.masked_laplace.tracers.clear()  # we will need to retrace because the matrix changes each step. This is not needed when JIT-compiling the physics.
    return smoke, velocity, pressure, obstacles, remaining_divergence


obstacles = OBSTACLES
for _ in view('smoke, velocity, pressure', play=True, namespace=globals()).range(warmup=1):
    smoke, velocity, pressure, obstacles, remaining_divergence = step(smoke, velocity, pressure, obstacles, dt=1)
