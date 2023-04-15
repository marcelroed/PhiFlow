from phi.jax.flow import *
from phi.vis import plot
from phi.math.backend import set_global_default_backend
from phi.geom._level_set import rotate_vec

set_global_default_backend(JAX)

# Signed distance fields for different 2D shapes

center_point = math.tensor([50, 50], channel('vector'))
center_point3d = math.tensor([50, 50, 50], channel('vector'))
radius = 40


def hypercube(x):
    return math.sum(math.abs(x - center_point), dim='vector') - 50


def hypersphere(x):
    return math.sqrt(math.sum(math.vec_squared(x - center_point), dim='vector')) - radius

@jit_compile
def hypersphere3d(x):
    return math.sqrt(math.sum(math.vec_squared(x - center_point3d), dim='vector')) - radius

def david_star(x):
    # If we consider the minimum of distances to two squares, one of them rotated by 45 degrees, we get a star
    vec_from_center = x - center_point
    straight_cube = math.sum(math.abs(vec_from_center), dim='vector') - 50

    # 45 degree rotation matrix
    rotation_matrix = math.tensor([[math.cos(math.pi / 4), -math.sin(math.pi / 4)], [math.sin(math.pi / 4), math.cos(math.pi / 4)]], channel(vector_out='x,y'), channel(vector_in='x,y'))

    rotated_vec_from_center = rotate_vec(rotation_matrix, vec_from_center)
    rotated_cube = math.sum(math.abs(rotated_vec_from_center), dim='vector') - 50
    distances = math.minimum(straight_cube, rotated_cube)
    return distances

coord_field2d = CenteredGrid(0, extrapolation.BOUNDARY, x=100, y=100, bounds=Box(x=100, y=100))

sdf = LevelSet(david_star, bounds=coord_field2d.bounds)

coord_pos = coord_field2d.elements.center

sdf_field = CenteredGrid(sdf, extrapolation.BOUNDARY, x=100, y=100, bounds=Box(x=100, y=100))

# boundary = math.abs(sdf_values) < 0.5
#
# sdf_values = math.where(boundary, math.nan, sdf_values)
view(sdf_field)
# exit()

# coord_field3d = CenteredGrid(0, extrapolation.BOUNDARY, x=10, y=10, z=10, bounds=Box(x=100, y=100, z=100))
sdf = LevelSet(hypersphere3d, bounds=Box(x=100, y=100, z=100))
# coord_pos = coord_field3d.elements.center
@jit_compile
def make_sdf_field():
    sdf_field = CenteredGrid(sdf, extrapolation.BOUNDARY, x=100, y=100, z=100, bounds=Box(x=100, y=100, z=100))
    return sdf_field

sdf_field = make_sdf_field()
# sdf_values = sdf.function(coord_pos)
# boundary = math.abs(sdf_values) < 0.5
# sdf_values = math.where(boundary, math.nan, sdf_values)
# plot(sdf_values)
print('Plotting')

view(sdf_field)
