import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gym
import mujoco
import numpy as np
import matplotlib.pyplot as plt

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definición de la Red de Actor
class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, action_dim)
        self.log_std = nn.Linear(512, action_dim)

    def forward(self, state, goal):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(next(self.parameters()).device)
        if not isinstance(goal, torch.Tensor):
            goal = torch.tensor(goal, dtype=torch.float32).to(next(self.parameters()).device)

        if state.dim() == 1:
            state = state.unsqueeze(0)  
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)  

        x = torch.cat([state, goal], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc3(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state, goal):
        mean, std = self.forward(state, goal)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)  
        return action, log_prob.sum(1, keepdim=True)

# Definición de la Red de Crítica
class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, goal, action):
        x = torch.cat([state, goal, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer con soporte para Hindsight Experience Replay
class HERReplayBuffer:
    def __init__(self, max_size, her_k=4):
        self.buffer = deque(maxlen=max_size)
        self.her_k = her_k

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        augmented_samples = []

        for state, action, reward, next_state, done, goal in samples:
            her_goals = [np.random.uniform(low=-1, high=1, size=3) for _ in range(self.her_k)]
            for new_goal in her_goals:
                new_reward = -np.linalg.norm(next_state[:3] - new_goal)
                augmented_samples.append((state, action, new_reward, next_state, done, new_goal))

        augmented_samples.extend(samples)
        states, actions, rewards, next_states, dones, goals = zip(*augmented_samples)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
                torch.tensor(goals, dtype=torch.float32))


# SAC con soporte para HER
class SAC:
    def __init__(self, state_dim, goal_dim, action_dim, max_action, lr=3e-4, gamma=0.95, tau=0.005, alpha=0.1):
        self.actor = Actor(state_dim, goal_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, goal_dim, action_dim).to(device)
        self.target_critic1 = Critic(state_dim, goal_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, goal_dim, action_dim).to(device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.replay_buffer = HERReplayBuffer(max_size=1_000_000)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.max_action = max_action

    def select_action(self, state, goal):
        action, _ = self.actor.sample(state, goal)
        return action.cpu().detach().numpy().flatten()

    def train(self, batch_size=256):
        states, actions, rewards, next_states, dones, goals = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, goals = (
            states.to(device), actions.to(device), rewards.to(device),
            next_states.to(device), dones.to(device), goals.to(device)
        )

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states, goals)
            target_q1 = self.target_critic1(next_states, goals, next_actions)
            target_q2 = self.target_critic2(next_states, goals, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * (torch.min(target_q1, target_q2) - self.alpha * next_log_probs)

        current_q1 = self.critic1(states, goals, actions)
        current_q2 = self.critic2(states, goals, actions)

        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        new_actions, log_probs = self.actor.sample(states, goals)
        q1 = self.critic1(states, goals, new_actions)
        q2 = self.critic2(states, goals, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_to_buffer(self, transition):
        self.replay_buffer.add(transition)


def get_state(data):
    state = np.concatenate([data.qpos, data.qvel])
    return state

def step_simulation(model, data):
    mujoco.mj_step(model, data)

def apply_action(data, action):
    action = np.clip(action, -1, 1)
    ctrl_min = model.actuator_ctrlrange[:, 0]
    ctrl_max = model.actuator_ctrlrange[:, 1]
    scaled_action = ctrl_min + (action + 1) * 0.5 * (ctrl_max - ctrl_min)
    data.ctrl[:] = scaled_action

def calculate_reward(data, target_position, tolerance=0.1):
    end_effector_id = 6
    end_effector_position = data.xpos[end_effector_id]
    distance_to_target = np.linalg.norm(end_effector_position - target_position)
    reward = -distance_to_target*100
    if distance_to_target < tolerance:
        reward += 100.0
    return reward

xml_path = "franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

state_dim = model.nq + model.nv
goal_dim = 3
action_dim = model.nu
max_action = 1.0

sac = SAC(state_dim, goal_dim, action_dim, max_action)

num_episodes = 120
max_steps = 100  
goal = np.array([-0.7, 0, 0.5])

rewards_history = []
distance_to_goal_history = []
losses_history = []
batch_size = 64

for episode in range(num_episodes):
    mujoco.mj_resetData(model, data)
    state = get_state(data)
    episode_reward = 0
    episode_distances = []
    actor_losses = []

    for step in range(max_steps):
        # Seleccionar acción
        action = sac.select_action(state, goal)
        if episode < 10:  # Exploración inicial
            action += np.random.normal(0, 0.1, size=action.shape)

        # Aplicar acción y avanzar simulación
        apply_action(data, action)
        step_simulation(model, data)

        # Obtener nuevo estado, recompensa, distancia, y verificar si termina
        next_state = get_state(data)
        reward = calculate_reward(data, goal)
        distance_to_goal = np.linalg.norm(data.xpos[6] - goal)
        done = step == max_steps - 1  # Marca fin de episodio si se alcanzan los pasos máximos

        # Guardar métricas de distancia
        episode_distances.append(distance_to_goal)

        # Agregar al buffer de replay
        sac.add_to_buffer((state, action, reward, next_state, done, goal))

        # Actualizar estado y recompensa acumulada
        state = next_state
        episode_reward += reward

        if done:
            break

    if done:
        break

