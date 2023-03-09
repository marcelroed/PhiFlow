"""
This tests a simple obstacle with force calculation and obstacle update.
We try compiling with JAX, since this has resulted in errors before.
"""

# Make sure we check for tracer leaks
import os
os.environ['JAX_CHECK_TRACER_LEAKS'] = '1'

from phi.flow import *
# from phi.jax.flow import *
from phi.torch.flow import *
from torch.autograd import detect_anomaly

DOMAIN = dict(x=100, y=100, bounds=Box(x=100, y=100))

DT = 1.0

center = DOMAIN['bounds'].center
radius = 20

def vec_l1(x):
    return math.sum(math.abs(x), dim='vector')

# The signed distance field for a circle in 2D
def signed_distance_field(x):
    return vec_l1(x) - radius

# arr = math.tensor([[0., 1.], [2., 3.]], batch('b'), channel('vector'))
# native = arr.native(['b', 'vector'])
# print(native)
# print(jax.vmap(signed_distance_field)(native))
# print(signed_distance_field(arr))
arr = math.tensor([[[0., 1.], [2., 3.]], [[0., 1.], [2., 3.]]], batch('x, y'), channel('vector'))
print(math.gradient(signed_distance_field, wrt='x', get_output=True)(x=arr))



obstacles = [
    #Obstacle(Box(x=(40, 60), y=(40, 60)), angular_velocity=0.05, mass=1e3)
    Obstacle(LevelSet(signed_distance_field, bounds=DOMAIN['bounds']))
]


# @jit_compile
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
