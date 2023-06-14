import time

from phi.torch.flow import *
import jax.numpy as jnp
import warp as wp
import warp.config
import warp.torch
from torch.utils.benchmark import Timer
from pytorch3d.ops.marching_cubes import marching_cubes as mc3d

backend_obj = backend.default_backend()
backend_obj.set_default_device(backend_obj.list_devices('GPU')[5])
device_name = 'cuda:5'

wp.config.print_launches = True
wp.init()

DOMAIN = dict(x=100, y=100, bounds=Box(x=100, y=100))

# Use Warp kernel for 3D marching cubes only
# marcher = wp.MarchingCubes(DOMAIN['x'] + 1, ny=DOMAIN['y'] + 1, nz=DOMAIN['z'] + 1, max_verts=10_000, max_tris=40_000,
#                            device='cuda:5')


def find_dimensions(verts, tris):
    # Find the dimension along which each vertex can change
    pass


def marching_cubes(corner_values: torch.Tensor):
    # nx, ny, nz = corner_values.shape
    w = wp.from_torch(corner_values)
    # with wp.ScopedStream(torch_stream):
    marcher.surface(w, 0.0)
    # wp.synchronize_stream(torch_stream)
    # print(marcher.verts.size, marcher.indices.size)
    verts, tris = wp.to_torch(marcher.verts)[:marcher.verts.size, :], wp.to_torch(marcher.indices)[
                                                                      :marcher.indices.size].reshape(-1, 3)
    # wp.stream_from_torch()
    # print(marcher.)
    return verts, tris


DT = 1.0
center = DOMAIN['bounds'].center
radius = 30


def vec_l2(x):
    return math.sqrt(math.sum(math.vec_squared(x), dim='vector'))


def sphere_sdf(x):
    return vec_l2(x - center) - radius


sphere = LevelSet(sphere_sdf, bounds=DOMAIN['bounds'])
grid = CenteredGrid(0, extrapolation.ZERO, **DOMAIN)
corner_values = sphere.function(grid.element_corners().center)

torch_corners = corner_values.native(corner_values.shape)
verts, tris = marching_cubes(torch_corners)
# results = []
# timer = Timer('marching_cubes(torch_corners)', globals=globals())
# print(timer.timeit(1000))
# torch_corners_b = torch_corners[None]
# timer2 = Timer('mc3d(torch_corners_b, 0.0)', globals=globals())
# print(timer2.timeit(1000))

# print(verts, tris)

# mask = CenteredGrid(corner_values < 0, bounds=DOMAIN['bounds'], extrapolation=extrapolation.ZERO)
# view('sphere', namespace=globals(), play=False)

print('Done')
exit()


# @jit_compile
def step(obstacles, velocity, frame):
    print(math.max(field.curl(velocity).data, 'x,y'))
    velocity = advect.mac_cormack(velocity, velocity, DT)
    velocity, pressure = fluid.make_incompressible(velocity, obstacles, Solve('CG-adaptive', 1e-5, 1e-5))
    obstacle_forces = fluid.pressure_to_obstacles(velocity, pressure, obstacles, dt=DT)
    obstacles = update_obstacles_forces(obstacles, obstacle_forces=obstacle_forces)
    fluid.masked_laplace.tracers.clear()  # we will need to retrace because the matrix changes each step. This is not needed when JIT-compiling the physics.
    global OBSTACLE_MASK
    OBSTACLE_MASK = CenteredGrid(obstacles[0].geometry, extrapolation.ZERO, **DOMAIN)
    return obstacles, velocity, pressure, OBSTACLE_MASK


for frame in view(velocity, OBSTACLE_MASK, namespace=globals(), framerate=10,
                  display=('velocity', 'pressure', 'OBSTACLE_MASK')).range():
    obstacles, velocity, pressure, OBSTACLE_MASK = step(obstacles, velocity, frame)
