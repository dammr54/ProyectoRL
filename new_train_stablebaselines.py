import numpy as np
import mujoco
import gymnasium as gym  # Gymnasium herramientas de entrenamiento de openAI
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from tqdm import tqdm

# ---------------- Configuración del entorno ------------------------
class FrankaPandaEnv(gym.Env):  # Heredar de gymnasium.Env
    def __init__(self, model, data, goal):
        super(FrankaPandaEnv, self).__init__()
        self.model = model
        self.data = data
        self.goal = np.array(goal, dtype=np.float32)  # Convertido a float32

        # Espacios de observaciones y acciones
        obs_dim = model.nq + model.nv
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        action_dim = model.nu
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        ctrl_range = self.model.actuator_ctrlrange
        scaled_action = np.clip(action, ctrl_range[:, 0], ctrl_range[:, 1])
        self.data.ctrl[:] = scaled_action

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = float(self._compute_reward(obs["achieved_goal"], self.goal))
        done = reward == 0
        return obs, reward, done, False, {}

    def _get_obs(self):
        return {
            "observation": np.concatenate([self.data.qpos[:], self.data.qvel[:]]).astype(np.float32),
            "achieved_goal": self.data.xpos[6].astype(np.float32),
            "desired_goal": self.goal,
        }

    def _compute_reward(self, achieved_goal, desired_goal, info=None):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return float(-(distance > 0.05).astype(np.float32))

# ---------------- Configuración de la simulación -------------------
xml_path = "franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
goal = [0.5, 0.5, 0.5]  # Meta deseada

env = FrankaPandaEnv(model, data, goal)
check_env(env)  # Verifica que el entorno sea compatible con Gymnasium

# Convierte el entorno a vectorizado para usar con Stable Baselines3
env = make_vec_env(lambda: env, n_envs=1)

# ---------------- Entrenamiento con SAC y HER ----------------------
# Define el buffer de replay con HER
her_kwargs = {
    "goal_selection_strategy": "future",  # Estrategia "future" para metas retrospectivas
    "n_sampled_goal": 4,  # Número de metas HER por transición
}

# Inicializa el modelo SAC con HER
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,  # Usando el buffer HER
    replay_buffer_kwargs=her_kwargs,  # Configuración de HER
    verbose=1,
    gamma=0.99,
    learning_rate=1e-4,
    buffer_size=1_000_000,
    tau=0.01,
    learning_starts=1000
)

# Define un callback para evaluación durante el entrenamiento
eval_callback = EvalCallback(
    env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=1000,
    deterministic=True,
)

# Entrena el modelo
model.learn(total_timesteps=500_000, callback=eval_callback)

# ---------------- Evaluación del Modelo ----------------------------
obs, _ = env.reset()
for _ in tqdm(range(1000)):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
