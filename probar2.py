import mujoco
from mujoco.glfw import glfw
import OpenGL.GL as gl
import numpy as np
from stable_baselines3 import SAC
import gymnasium as gym
from gymnasium import spaces

# Cargar el modelo MuJoCo
xml_path = "franka_fr3_dual/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10000)
mujoco.mjv_defaultOption(opt)

# Configuración inicial de la cámara
cam.azimuth = 90
cam.elevation = -8
cam.distance = 4
cam.lookat = np.array([0.0, 0.0, 1])


# copia de una clase ya definida en otro archivo
class MujocoEnvWithGoals(gym.Env):
    def __init__(self, model, data, goal, max_steps=1000):
        super(MujocoEnvWithGoals, self).__init__()
        self.model = model
        self.data = data
        self.goal = goal
        self.step_count = 0
        self.max_steps = max_steps
        self.all_distances = []
        self.all_rewards = []

        obs_dim = model.nq + model.nv
        self.observation_space = spaces.Dict({
            "observation": spaces.box.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
            "achieved_goal": spaces.box.Box(low=-np.inf, high=np.inf, shape=(3,)),
            "desired_goal": spaces.box.Box(low=-np.inf, high=np.inf, shape=(3,))
        })
        self.action_space = spaces.box.Box(low=-1.0, high=1.0, shape=(model.nu,))
        self.tolerance = 0.01

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self.compute_reward(self.data, self.goal)
        distance_to_target_actual = self.all_distances[-1] if self.all_distances else np.inf
        self.all_rewards.append(reward)
        self.all_distances.append(distance_to_target_actual)
        terminated = distance_to_target_actual <= self.tolerance
        truncated = self.step_count >= self.max_steps
        info = {"is_success": float(terminated)}
        self.step_count += 1
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.all_distances = []
        self.all_rewards = []
        return self._get_obs(), {}

    def _get_obs(self):
        achieved_goal = self.data.xpos[6]
        observation = np.concatenate([self.data.qpos, self.data.qvel])
        return {"observation": observation,
                "achieved_goal": achieved_goal,
                "desired_goal": self.goal}

    def compute_reward(self, data, target_position, info=None):
        end_effector_position = self.data.xpos[6]
        distance_to_target = np.linalg.norm(end_effector_position - target_position)
        self.all_distances.append(distance_to_target)
        return -(distance_to_target)

# Cargar el modelo entrenado
trained_model_path = "her_sac_trained_model.zip"
goal = np.array([0.7, -0.5, 0.5])
env_obs_space = model.nq + model.nv
env = MujocoEnvWithGoals(model, data, goal)
sac_model = SAC.load(trained_model_path, env=env)

# Inicializar GLFW
def glfw_init():
    width, height = 1280, 720
    window_name = "Simulación MuJoCo con Modelo SAC"

    if not glfw.init():
        raise Exception("No se pudo inicializar el contexto OpenGL")

    window = glfw.create_window(width, height, window_name, None, None)
    if not window:
        glfw.terminate()
        raise Exception("No se pudo inicializar la ventana GLFW")

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    return window

# Variables para el movimiento del mouse
mouse_lastx, mouse_lasty = 0, 0
mouse_button_left, mouse_button_middle, mouse_button_right = False, False, False

# Callbacks para interacción con el mouse
def mouse_button(window, button, act, mods):
    global mouse_button_left, mouse_button_middle, mouse_button_right
    mouse_button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    mouse_button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    mouse_button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

def mouse_scroll(window, xoffset, yoffset):
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)

def mouse_move(window, xpos, ypos):
    global mouse_lastx, mouse_lasty
    dx, dy = xpos - mouse_lastx, ypos - mouse_lasty
    mouse_lastx, mouse_lasty = xpos, ypos

    if not (mouse_button_left or mouse_button_middle or mouse_button_right):
        return

    width, height = glfw.get_window_size(window)
    if mouse_button_right:
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_V, dx / height, dy / height, scene, cam)
    if mouse_button_left:
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx / height, dy / height, scene, cam)

# Simulación principal
def main():
    window = glfw_init()
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, mouse_scroll)
    glfw.set_cursor_pos_callback(window, mouse_move)

    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    # Resetear entorno
    mujoco.mj_resetData(model, data)
    data.qpos[:] = np.zeros_like(data.qpos)
    mujoco.mj_forward(model, data)

    obs = np.concatenate([data.qpos, data.qvel])
    done = False

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Obtener acción del modelo entrenado
        action, _ = sac_model.predict({"observation": obs, "achieved_goal": data.xpos[6], "desired_goal": goal})
        data.ctrl[:] = action

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step Reward: {reward}, Info: {info}")

        # Paso de simulación
        mujoco.mj_step(model, data)
        obs = np.concatenate([data.qpos, data.qvel])

        # Renderizar escena
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)

    glfw.terminate()

# Ejecutar la simulación
if __name__ == "__main__":
    main()
