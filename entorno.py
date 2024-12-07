import mujoco
from mujoco.glfw import glfw
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
#from gymnasium.utils.env_checker import check_env


# ---------------- Configuración del Modelo ------------------------
xml_path = "franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)  # Cargar el modelo MuJoCo
data = mujoco.MjData(model)  # Crear datos dinámicos asociados

# Configuración inicial de la cámara
cam = mujoco.MjvCamera()
cam.azimuth = 90
cam.elevation = -8
cam.distance = 2
cam.lookat = np.array([0.0, 0.0, 1.0])

# Opciones de visualización
opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)
scene = mujoco.MjvScene(model, maxgeom=10000)

# Renderizado
def glfw_init():
    width, height = 1280, 720
    if not glfw.init():
        print("No se pudo inicializar GLFW")
        exit(1)
    window = glfw.create_window(width, height, "Simulación Franka", None, None)
    if not window:
        glfw.terminate()
        print("No se pudo crear la ventana GLFW")
        exit(1)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    return window

window = glfw_init()
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# ---------------- Definición del Entorno ------------------------
class FrankaPandaEnv(gym.Env):
    def __init__(self, model, data):
        super(FrankaPandaEnv, self).__init__()
        self.model = model
        self.data = data

        # Espacio de observaciones (posiciones + velocidades articulares)
        obs_dim = model.nq + model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Espacio de acciones (control de actuadores)
        action_dim = model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def reset(self):
        # Reiniciar el estado del modelo
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        # Escalar las acciones dentro de los límites de los actuadores
        ctrl_range = self.model.actuator_ctrlrange
        scaled_action = np.clip(action, ctrl_range[:, 0], ctrl_range[:, 1])
        self.data.ctrl[:] = scaled_action

        # Avanzar la simulación
        mujoco.mj_step(self.model, self.data)

        # Obtener observaciones
        obs = self._get_obs()

        # Recompensa: minimizar la distancia al objetivo
        goal_pos = np.array([0.5, 0.5, 0.5])  # Posición objetivo
        ee_pos = self.data.site_xpos[0]  # Posición de la herramienta final
        reward = -np.linalg.norm(goal_pos - ee_pos)  # Recompensa negativa

        # Determinar si el episodio termina
        done = False

        return obs, reward, done, {}

    def _get_obs(self):
        # Combinar posiciones y velocidades
        return np.concatenate([self.data.qpos[:], self.data.qvel[:]])

    def render(self, mode='human'):
        # Renderizar la simulación
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        mujoco.mjv_updateScene(self.model, self.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

# ---------------- Entrenamiento del Agente ------------------------
# Crear el entorno
env = FrankaPandaEnv(model, data)

# Verificar el entorno
check_env(env)

# Crear el agente SAC
agent = SAC("MlpPolicy", env, verbose=1, learning_rate=0.001, gamma=0.99)

# Entrenar el agente
agent.learn(total_timesteps=50000)

# Guardar el modelo entrenado
agent.save("sac_franka_panda")

# ---------------- Evaluación del Modelo Entrenado -----------------
# Cargar el modelo entrenado
agent = SAC.load("sac_franka_panda")

# Ejecutar el modelo en el entorno con renderizado
obs = env.reset()
for _ in range(1000):
    env.render()
    action, _ = agent.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()

# Terminar GLFW al final
glfw.terminate()
