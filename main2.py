import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('franka_fr3/scene.xml')
data = mujoco.MjData(m)

cam = mujoco.MjvCamera()  # Crea la cámara abstracta de MuJoCo
opt = mujoco.MjvOption()                    # Visualization options


cam.azimuth = 89.608063  # Ángulo azimutal de la cámara
cam.elevation = -11.588379  # Ángulo de elevación
cam.distance = 20.0  # Distancia de la cámara
cam.lookat = np.array([0.0, 0.0, 1.5])  # Punto de interés

mujoco.mjv_defaultOption(opt)
scene = mujoco.MjvScene(m, maxgeom=10000)

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)