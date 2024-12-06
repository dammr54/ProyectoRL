import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

torch.autograd.set_detect_anomaly(True)

# Definimos el device (GPU si está disponible, si no, usamos la CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definimos los parámetros de entrenamiento
BUFFER_SIZE = 1000000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 3e-4
ALPHA = 0.2  # Parámetro de temperatura para la entropía

# Cargar el modelo XML de MuJoCo (asegúrate de que la ruta del archivo XML sea correcta)
model_path = "franka_fr3/fr3.xml"  # Asegúrate de tener el archivo XML de MuJoCo

# Cargar el modelo MuJoCo
model = mujoco.MjModel.from_xml_path(model_path)

# Crear el objeto MjData para simular
data = mujoco.MjData(model)

# Definir el entorno (código simplificado)
class MuJoCoEnv:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(self.model)  # Usar MjData en lugar de MjSim

    def reset(self):
        # Restablecemos la simulación
        mujoco.mj_resetData(self.model, self.data)
        return self.get_state()

    def step(self, action):
        # Aplicamos las acciones en el control
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # Obtenemos el nuevo estado
        next_state = self.get_state()

        # Calculamos la recompensa (en este caso un ejemplo básico)
        reward = -np.linalg.norm(next_state[:3])  # Ejemplo: minimizar la distancia a un objetivo (cambiar según el objetivo)
        
        # Verificamos si ha terminado
        done = False  # Implementar lógica de finalización según el problema

        return next_state, reward, done, {}

    def get_state(self):
        # Obtenemos la posición (qpos) y la velocidad (qvel) del robot
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

# Red neuronal para el Actor-Crítico
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Crítico
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action = self.actor(state)
        value = self.critic(torch.cat([state, action], dim=-1))
        return action, value

# Agente SAC (Soft Actor-Critic)
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim):
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=LEARNING_RATE)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action, _ = self.actor_critic(state)
        return action.cpu().detach().numpy()

    def train(self, state, action, reward, next_state, done):
        # Entrenamiento basado en los valores de la acción y el crítico
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor([done]).to(device)

        _, current_value = self.actor_critic(state)
        _, next_value = self.actor_critic(next_state)

        target_value = reward + (1 - done) * GAMMA * next_value
        # Entrenamiento del crítico
        critic_loss = F.mse_loss(current_value, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)  # Usar retain_graph=True aquí para mantener el gráfico

        self.critic_optimizer.step()

        # Entrenamiento del actor
        actor_loss = -current_value.mean()  # Entropía y Q-value del actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()  # Ya no necesitamos retain_graph aquí porque el gráfico se retiene de la retropropagación anterior
        self.actor_optimizer.step()

# Entrenamiento
def train():
    env = MuJoCoEnv(model)  # Inicializa el entorno con el modelo
    agent = SACAgent(state_dim=14, action_dim=7, hidden_dim=256)  # 14 dimensiones de estado, 7 acciones (grados de libertad)
    
    for episode in range(1000):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        print(f"Episode {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    train()
