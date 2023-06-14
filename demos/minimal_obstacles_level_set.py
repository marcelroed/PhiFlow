"""
This tests a simple obstacle with force calculation and obstacle update.
We try compiling with JAX, since this has resulted in errors before.
"""

# Make sure we check for tracer leaks
import os

# os.environ['JAX_CHECK_TRACER_LEAKS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# from phi.jax.flow import *
from phi.torch.flow import *

DOMAIN = dict(x=100, y=100, bounds=Box(x=100, y=100))

DT = 0.01

center = DOMAIN['bounds'].center
radius = 5


def vec_l1(x):
    return math.sum(math.abs(x - center), dim='vector')


def square(x):
    return vec_l1(x) - radius


def circle(x):
    result = math.vec_length(x - center, 'vector') - radius
    return result


# arr = math.tensor([[0., 1.], [2., 3.]], batch('b'), channel('vector'))
# native = arr.native(['b', 'vector'])
# print(native)
# print(jax.vmap(signed_distance_field)(native))
# print(signed_distance_field(arr))
# arr = math.tensor([[[0., 1.], [2., 3.]], [[0., 1.], [2., 3.]]], batch('x, y'), channel('vector'))
# print(math.gradient(signed_distance_field, wrt='x', get_output=True)(x=arr))
#
# obstacles = [
#     # Obstacle(Box(x=(40, 60), y=(40, 60)), angular_velocity=0.05, mass=1e3)
#     Obstacle(LevelSet(signed_distance_field, bounds=DOMAIN['bounds']))
# ]

function = square
level_set = LevelSet(function, bounds=DOMAIN['bounds'])
# stationary_box = phi.geom._transform.RotatedGeometry(Box(x=(0, 100), y=(0, 20)), angle=45)
obstacles = [Obstacle(level_set, mass=1e3),  # Moving box
             # Obstacle(stationary_box, mass=1e100, moment_of_inertia=1e100)  # Stationary wall
]

# @jit_compile
def step(obstacles, velocity):
    print('Advection start')
    velocity = advect.mac_cormack(velocity, velocity, dt=DT)
    print('Incompressible start')
    velocity, pressure = fluid.make_incompressible(velocity, obstacles, solve=Solve('auto'))
    print('Ending')
    obstacle_forces = fluid.pressure_to_obstacles(velocity=None, pressure=pressure, obstacles=obstacles, dt=DT)
    # print(obstacle_forces)
    new_obstacles = update_obstacles_forces(obstacles, obstacle_forces, dt=DT)
    # print(new_obstacles)
    return velocity, pressure, new_obstacles


pressure = CenteredGrid(0, extrapolation.ZERO, **DOMAIN)
velocity = StaggeredGrid(0, extrapolation=extrapolation.combine_sides(x=(math.tensor([1., 0.], channel(vector='x,y')), extrapolation.ZERO_GRADIENT), y=(extrapolation.ZERO_GRADIENT, extrapolation.ZERO_GRADIENT)), **DOMAIN)
# cell_values = math.dot(pressure.elements.center, 'vector', math.tensor([1., 2.], channel(vector='x,y')) / 10, 'vector')
# pressure = CenteredGrid(cell_values, extrapolation.ZERO, **DOMAIN)
print(pressure.values.native(pressure.values.shape))

obstacle_geometry = obstacles[0].geometry
# obstacle_mask = CenteredGrid(obstacle_geometry, extrapolation.ZERO, **DOMAIN)


i = 0
for s in view('pressure, velocity, obstacle_geometry', namespace=globals()).range(warmup=1):
    # print(i := i + 1)
    velocity, pressure, obstacles = step([], velocity=velocity)
    # obstacle_geometry = obstacles[0].geometry
    # obstacle_2_geometry = obstacles[1].geometry
    # obstacle_mask = CenteredGrid(obstacle_geometry, extrapolation.ZERO, **DOMAIN)
    # print(s)
