# import gym
import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from stable_baselines3 import SAC, HerReplayBuffer
# from stable_baselines3.common.buffers import HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

# ---------------- Adaptar entorno a Gym ----------------
class MujocoEnvWithGoals(gym.Env):
    def __init__(self, model, data, goal):
        super(MujocoEnvWithGoals, self).__init__()
        self.model = model
        self.data = data
        self.goal = goal
        self.step_count = 0
        self.max_steps = 1000

        # Espacios de observación y acción
        obs_dim = model.nq + model.nv
        # self.observation_space = spaces.Dict({
        #     "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
        #     "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        #     "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        # })
        # self.observation_space = obs_dim
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(model.nu,))
        # self.action_space = model.nu
        obs_dim = model.nq + model.nv
        self.observation_space = spaces.Dict({
            "observation": spaces.box.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
            "achieved_goal": spaces.box.Box(low=-np.inf, high=np.inf, shape=(3,)),
            "desired_goal": spaces.box.Box(low=-np.inf, high=np.inf, shape=(3,))})
        self.action_space = spaces.box.Box(low=-1.0, high=1.0, shape=(model.nu,))

        self.tolerance = 0.8  # Inicializar tolerancia para la recompensa

    # def step(self, action):
    #     self.data.ctrl[:] = action  # Asignar directamente la acción al controlador
    #     mujoco.mj_step(self.model, self.data)
    #     obs = self._get_obs()
    #     # reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None)
    #     reward = self.compute_reward(self.data, self.goal)
    #     done = reward >= -self.tolerance
    #     info = {"is_success": float(done)}
    #     return obs, reward, done, done, info

    def step(self, action):
        self.data.ctrl[:] = action  # Asignar directamente la acción al controlador
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self.compute_reward(self.data, self.goal)
    
        terminated = False
        truncated = False
    
        if reward >= -self.tolerance:
            terminated = True
    
        # Aquí podemos considerar un criterio adicional para truncar si el episodio excede un límite de pasos
        # Por ejemplo, si max_steps es un parámetro del entorno:
        if self.step_count >= self.max_steps:  # Asume step_count lleva cuenta de los pasos dados
            truncated = True
    
        info = {"is_success": float(terminated)}
        self.step_count += 1
    
        return obs, reward, terminated, truncated, info


    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        achieved_goal = self.data.xpos[6]
        observation = np.concatenate([self.data.qpos, self.data.qvel])
        return {"observation": observation,
                "achieved_goal": achieved_goal,
                "desired_goal": self.goal}
        # return {
        #     "observation": observation,
        #     "achieved_goal": achieved_goal,
        #     "desired_goal": self.goal
        # }

    # def compute_reward(self, achieved_goal, desired_goal, info):
    #     distance = np.linalg.norm(achieved_goal - desired_goal)
    #     return -distance
    def compute_reward(self, data, target_position, info=None):
        end_effector_position = self.data.xpos[6]
        distance_to_target = np.linalg.norm(end_effector_position - target_position)
        return -distance_to_target

# ---------------- Configuración y Entrenamiento ----------------
xml_path = "franka_fr3_dual/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
# max_steps = 1000

goal = np.array([0.7, -0.5, 0.5])

# Crear el entorno
env = MujocoEnvWithGoals(model, data, goal)

# Configurar el modelo SAC con los parámetros correctos de HerReplayBuffer
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        # buffer_size=1000000,  # Tamaño máximo del búfer
        # observation_space=env.observation_space,
        # action_space=env.action_space,
        # device="auto",  # Detectar dispositivo automáticamente
        # n_envs=1,  # Número de entornos paralelos (1 en este caso)
        # optimize_memory_usage=False,  # Variante eficiente de memoria
        handle_timeout_termination=True,  # Manejar terminación por límite de tiempo
        n_sampled_goal=4,  # Número de objetivos virtuales por transición
        goal_selection_strategy=GoalSelectionStrategy.FUTURE,  # Estrategia de selección de objetivos
        copy_info_dict=False  # No copiar info para computar recompensas
    ),
    verbose=1,
)

# Entrenamiento del modelo
num_episodes = 100
max_steps = 1000
try:
    model.learn(total_timesteps=num_episodes * max_steps)
    model.save("her_sac_trained_model")
except:
    model.save("her_sac_trained_model")


# Guardar el modelo
model.save("her_sac_ANAIS_trained_model")

# Evaluación del modelo
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Recompensa media: {mean_reward}, desviación estándar: {std_reward}")

