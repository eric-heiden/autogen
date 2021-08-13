import pytinydiffsim_ad as dp
import autogen as ag
import numpy as np
import math

TIME_STEPS = 20

def func(input_tau):
  world = dp.TinyWorld()
  world.friction = dp.ADScalar(1.0)

  urdf_parser = dp.TinyUrdfParser()
  urdf_data = urdf_parser.load_urdf("/root/tiny-differentiable-simulator/data/cartpole.urdf")
  print("robot_name=",urdf_data.robot_name)
  # b2vis = meshcat_utils_dp.convert_visuals(urdf_data, "~/tiny-differentiable-simulator/data/laikago/laikago_tex.jpg", vis, "../../data/laikago/")
  is_floating=False
  mb = dp.TinyMultiBody(is_floating)
  urdf2mb = dp.UrdfToMultiBody2()
  res = urdf2mb.convert2(urdf_data, world, mb)

  dt = dp.ADScalar(1./1000.)
  cost_output = np.ones(TIME_STEPS, dtype=ag.ADScalar)

  for i in range(TIME_STEPS):
    # print(type(input_tau[0]))

    # convert to tds type
    mb.tau[0] = input_tau[i].value()
    dp.forward_dynamics(mb, world.gravity)
    dp.integrate_euler(mb, dt)

    # convert to ag type
    pole_cost = math.sin(mb.q[1].value()) ** 2
    cart_cost = (mb.q[0].value() / 2.4) ** 2
    total_cost = pole_cost + cart_cost
    cost_output[i] = ag.ADScalar(total_cost)
  print('cost arr=', cost_output)
  return cost_output

f = ag.trace(func, [1.0] * TIME_STEPS)
gen = ag.GeneratedCppAD(f)

x = [2.0] * TIME_STEPS
y = f.forward(x)
print("y = ", y)
J = f.jacobian(x)
print("j = ", J)