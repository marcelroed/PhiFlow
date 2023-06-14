from phi.torch.flow import *
import torch._dynamo.config as dc

dc.verbose = True


DOMAIN = dict(x=64, y=64, bounds=Box(x=64, y=64))

top_and_bottom_vel = math.tensor([0, 1], channel(vector='x,y'))
velocity = StaggeredGrid(0, extrapolation=extrapolation.combine_sides(x=extrapolation.ZERO,
                                                                      y=(top_and_bottom_vel, top_and_bottom_vel)),
                         **DOMAIN)

def step(velocity):
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
    velocity, pressure = fluid.make_incompressible(velocity)
    return velocity

def loss_val(velocity: StaggeredGrid):
    velocity = step(velocity)
    return math.sum(velocity.values[0]) + math.sum(velocity.values[1])



grad_func = functional_gradient(loss_val, wrt='velocity', get_output=True)

grad_func_jit = grad_func

value, gradient = grad_func_jit(velocity)
print(gradient, value)