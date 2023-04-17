"""
We attempt to implement Box and Sphere both in terms of PhiFlow primitives and as LevelSets.
The LevelSet implementation should give the same results as using the primitives directly.
"""

# Make sure we check for tracer leaks
import os

os.environ['JAX_CHECK_TRACER_LEAKS'] = '1'

# from phi.jax.flow import *
from phi.torch.flow import *
import jax

backend_obj = backend.default_backend()
backend_obj.set_default_device(backend_obj.list_devices('GPU')[5])

DOMAIN = dict(x=100, y=100, z=100, bounds=Box(x=100, y=100, z=100))

DT = 1.0

center = DOMAIN['bounds'].center
radius = 10


def vec_l1(x):
    return math.sum(math.abs(x), dim='vector')


def vec_l2(x):
    return math.sqrt(math.sum(math.vec_squared(x), dim='vector'))


# The signed distance field for a circle in 2D
def circle_sdf(x):
    return vec_l2(x - center) - radius


def box_sdf(x):
    # Center the box in the domain
    x = x - center
    # # Rotate the box by 45 degrees
    # x = math.rotate_vector(x, angle=PI / 4)
    return vec_l1(x) - radius * math.sqrt(2)

# arr = math.tensor([[0., 1.], [2., 3.]], batch('b'), channel('vector'))
# native = arr.native(['b', 'vector'])
# print(native)
# print(jax.vmap(signed_distance_field)(native))
# print(signed_distance_field(arr))

# Make sure gradients work
# arr = math.tensor([[[0., 1.], [2., 3.]], [[0., 1.], [2., 3.]]], batch('x, y'), channel('vector'))
# print(math.gradient(box_sdf, wrt='x', get_output=True)(x=arr))

box_lower = math.tensor([40, 40, 40], channel(vector='x,y,z'))
box_upper = math.tensor([60, 60, 60], channel(vector='x,y,z'))

obstacles = {
    'standard': {
        'box': Obstacle(Box(lower=box_lower, upper=box_upper), angular_velocity=0, mass=1e3),
        # 'circle': Obstacle(Sphere(center=center, radius=radius), angular_velocity=0.05, mass=1e3),
    },
    'levelset': {
        'circle': Obstacle(LevelSet(circle_sdf, bounds=DOMAIN['bounds'])),
        # 'box': Obstacle(LevelSet(box_sdf, bounds=DOMAIN['bounds']), mass=1e3),
    }
}

# Define a pressure field with a constant gradient in the normal_vec direction
normal_vec = math.vec_normalize(math.tensor([1, 1, 1], channel(vector='x,y,z')), vec_dim='vector')
grid = CenteredGrid(0, extrapolation.ZERO, **DOMAIN)
pos = grid.elements.center
pressure = CenteredGrid(math.dot(pos, pos.shape.channel, normal_vec, normal_vec.shape.channel), extrapolation.BOUNDARY, **DOMAIN)


def get_obstacle_forces(obstacle, pressure_field):
    force, = fluid.pressure_to_obstacles(velocity=None, obstacles=[obstacle], pressure=pressure_field, dt=None)
    return force


# In a constant pressure gradient field with gradient normal_vec, the force on the box should be - volume_displacement * normal_vec
# We expect zero torque
expected_force_box = - obstacles['standard']['box'].geometry.volume * normal_vec

corners = pressure.element_corners()

# standard_box = CenteredGrid(obstacles['standard']['box'].geometry, extrapolation.ZERO, **DOMAIN)
# levelset_box = CenteredGrid(obstacles['levelset']['box'].geometry, extrapolation.ZERO, **DOMAIN)

# # Visualize the obstacle geometries
# view('standard_box,levelset_box', namespace=globals())
# exit()

results = jax.tree_map(lambda o: get_obstacle_forces(o, pressure), obstacles)

print(results, expected_force_box)

# @jit_compile
def step(obstacles, pressure):
    obstacle_forces = fluid.pressure_to_obstacles(velocity=None, pressure=pressure, obstacles=obstacles, dt=DT)
    obstacles = update_obstacles_forces(obstacles, obstacle_forces, dt=DT)
    return obstacles


pressure = CenteredGrid(0, extrapolation.ZERO, **DOMAIN)
obstacle_geometry = obstacles[0].geometry
obstacle_mask = CenteredGrid(obstacle_geometry, extrapolation.ZERO, **DOMAIN)
i = 0
for s in view('pressure, obstacle_mask', namespace=globals()).range(warmup=1):
    print(i := i + 1)
    obstacles = step(obstacles, pressure=pressure)
    obstacle_geometry = obstacles[0].geometry
    obstacle_mask = CenteredGrid(obstacle_geometry, extrapolation.ZERO, **DOMAIN)
    print(s)
