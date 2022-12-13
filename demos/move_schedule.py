from phi.flow import *
import jax.numpy as jnp
from phi.jax import *


DOMAIN = dict(x=100, y=100, bounds=Box(x=100, y=100))
DT = 1.0
obstacle = Obstacle(Box(x=(40, 60), y=(40, 60)), angular_velocity=0.05)
OBSTACLE_MASK = CenteredGrid(obstacle.geometry, 0, **DOMAIN)  # to show in user interface
velocity = StaggeredGrid(0,
                         extrapolation.ZERO,
                         # extrapolation.ConstantExtrapolation(math.tensor([1., 0.], channel(vector='x,y'))),
                         **DOMAIN)

original_geometry = obstacle.geometry
pressure = None

def position_schedule(t):
    return math.tensor([jnp.sin(t) * 30, jnp.cos(t) * 30]), t * 3

# @jit_compile
def step(obstacle, velocity, frame):
    pos, rot = position_schedule(frame / 10)
    print(pos, rot)
    previous_pos, previous_rot = position_schedule((frame - 1) / 10)
    obstacle_velocity = (pos - previous_pos) / DT
    obstacle_angular_velocity = (rot - previous_rot) / DT
    velocity = advect.mac_cormack(velocity, velocity, DT)
    obstacle = obstacle.copied_with(geometry=original_geometry.shifted(math.tensor(pos)).rotated(rot), velocity=obstacle_velocity, angular_velocity=obstacle_angular_velocity)  # rotate bar
    print(math.max(field.curl(velocity).data, 'x,y'))
    velocity, pressure = fluid.make_incompressible(velocity, (obstacle,), Solve('CG-adaptive', 1e-5, 1e-5))
    fluid.masked_laplace.tracers.clear()  # we will need to retrace because the matrix changes each step. This is not needed when JIT-compiling the physics.
    OBSTACLE_MASK = CenteredGrid(obstacle.geometry, extrapolation.ZERO, **DOMAIN)
    return obstacle, velocity, pressure, OBSTACLE_MASK


for frame in view(velocity, OBSTACLE_MASK, namespace=globals(), framerate=10, display=('velocity', 'OBSTACLE_MASK', 'pressure')).range():
    obstacle, velocity, pressure, OBSTACLE_MASK = step(obstacle, velocity, frame)
