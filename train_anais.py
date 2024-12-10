import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import mujoco

device_ = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device_}")
device = torch.device(device_)

# ---------------- Definición de Redes ------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state, goal):
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
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)


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

# ---------------- Hindsight Experience Replay (HER) ----------------
class HERReplayBuffer:
    def __init__(self, max_size, her_k=10):
        self.buffer = deque(maxlen=max_size)
        self.her_k = her_k

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        augmented_samples = []

        for state, action, reward, next_state, done, goal in samples:
            her_goals = [next_state[:3] for _ in range(self.her_k)]
            for new_goal in her_goals:
                new_reward = -np.linalg.norm(next_state[:3] - new_goal)
                augmented_samples.append((state, action, new_reward, next_state, done, new_goal))

        augmented_samples.extend(samples)
        augmented_samples = np.array(augmented_samples, dtype=object)
        states, actions, rewards, next_states, dones, goals = zip(*augmented_samples)

        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(np.array(actions), dtype=torch.float32),
                torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(goals), dtype=torch.float32))

# ---------------- Soft Actor-Critic (SAC) ----------------
class SAC:
    def __init__(self, state_dim, goal_dim, action_dim, max_action, lr=3e-4, gamma=0.99, tau=0.005, alpha="auto"):
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
        self.max_action = max_action

        # Si alpha es "auto", inicializamos log_alpha para aprenderlo
        if alpha == "auto":
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.target_entropy = -action_dim  # Entropía objetivo para el espacio de acciones
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

    def select_action(self, state, goal):
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        goal = torch.tensor(goal, dtype=torch.float32).to(device).unsqueeze(0)
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

        # Actualizamos el valor de alpha solo si está en "auto"
        if hasattr(self, "log_alpha"):
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Actualización de las redes objetivo (target networks)
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_to_buffer(self, transition):
        self.replay_buffer.add(transition)


# ---------------- Entrenamiento del Agente ----------------
def get_state(data):
    return np.concatenate([data.qpos, data.qvel])

def calculate_reward(data, target_position, tolerance=0.1):
    end_effector_position = data.xpos[6]
    distance_to_target = np.linalg.norm(end_effector_position - target_position)
    reward = -distance_to_target * 1
    # print(f"distance to target: {distance_to_target}")
    if distance_to_target < tolerance:
        reward += 1
        print("omg un reward")
    return reward

xml_path = "franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

state_dim = model.nq + model.nv
goal_dim = 3
action_dim = model.nu
max_action = 1.0

sac = SAC(state_dim, goal_dim, action_dim, max_action)

num_episodes = 150
max_steps = 200
goal = np.array([0.5, 0.5, 0.5])

for episode in range(num_episodes):
    mujoco.mj_resetData(model, data)
    state = get_state(data)
    episode_reward = 0

    for step in range(max_steps):
        action = sac.select_action(state, goal)
        apply_action = np.clip(action, -1, 1)  # Escalar acción

        # Aplica la acción al entorno y avanza la simulación
        data.ctrl[:] = apply_action
        mujoco.mj_step(model, data)

        # Calcula el nuevo estado y la recompensa
        next_state = get_state(data)
        reward = calculate_reward(data, goal)
        done = step == max_steps - 1

        # Agrega la transición al buffer de experiencia
        sac.add_to_buffer((state, apply_action, reward, next_state, done, goal))

        # Actualiza el estado actual y acumula la recompensa
        state = next_state
        episode_reward += reward

        # Entrena el modelo si hay suficientes datos en el buffer
        if len(sac.replay_buffer.buffer) > 256:
            sac.train(batch_size=256)

        # Termina el episodio si está completo
        if done:
            break

    # Imprime la recompensa total por episodio
    print(f"Episodio {episode + 1}, Recompensa Total: {episode_reward:.2f}")

    # Guarda el modelo cada 50 episodios
    if (episode + 1) % 50 == 0:
        torch.save({
            "actor": sac.actor.state_dict(),
            "critic1": sac.critic1.state_dict(),
            "critic2": sac.critic2.state_dict(),
            "actor_optimizer": sac.actor_optimizer.state_dict(),
            "critic1_optimizer": sac.critic1_optimizer.state_dict(),
            "critic2_optimizer": sac.critic2_optimizer.state_dict(),
        }, f"sac_checkpoint_{episode + 1}.pth")

