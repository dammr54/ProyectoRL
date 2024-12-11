import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import mujoco
import funciones_pickle as fpickle
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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
    def __init__(self, state_dim, goal_dim, action_dim, max_action, lr=1e-4, gamma=0.99, tau=0.01, alpha=0.1):
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


def calculate_reward(data, target_position, tolerance, tope_tolerance=0.05, max_tolerance_change=0.05, max_tolerance=0.6):
    # Posición del efector final
    end_effector_position = data.xpos[6]  # Ajustar el índice según tu modelo
    distance_to_target = np.linalg.norm(end_effector_position - target_position)

    # Calcula la orientación deseada
    desired_orientation = calculate_desired_orientation(end_effector_position, target_position)

    # Premiar o penalizar según la distancia al objetivo
    if distance_to_target <= tolerance:
        reward = 10 * (1 / (distance_to_target + 1e-6))  # Recompensa positiva inversamente proporcional a la distancia
        if distance_to_target < tope_tolerance:
            reward += 20  # Bonificación adicional si está muy cerca del objetivo
    else:
        reward = -10 * distance_to_target  # Penalización proporcional a la distancia

    # Penalización por esfuerzo (torque de los actuadores)
    torque_effort = np.sum(np.abs(data.ctrl))  # Magnitud del control aplicado
    reward -= 0.01 * torque_effort

    # Penalización por desviación de orientación
    current_orientation = data.xmat[6][:3]  # Obtener la orientación actual del efector final
    orientation_error = np.linalg.norm(current_orientation - desired_orientation)
    reward -= 0.1 * orientation_error

    # Penalización por desviación de las articulaciones
    joint_positions = data.qpos[:model.nq]  # Posiciones actuales de las articulaciones
    joint_velocity = data.qvel[:model.nv]  # Velocidades actuales de las articulaciones
    desired_joint_positions = np.zeros(model.nq)  # Ajustar según el estado objetivo de las articulaciones
    joint_position_error = np.linalg.norm(joint_positions - desired_joint_positions)
    reward -= 0.05 * joint_position_error

    # Actualización de tolerancia
    new_tolerance = max(tolerance - (tolerance - distance_to_target), tope_tolerance)  # Reduce la tolerancia
    new_tolerance = min(new_tolerance, tolerance + max_tolerance_change)  # Limita el incremento de tolerancia
    new_tolerance = min(new_tolerance, max_tolerance)  # Limita la tolerancia máxima a max_tolerance

    # Clipping de la recompensa
    reward = np.clip(reward, -50, 50)

    return reward, new_tolerance, distance_to_target


def calculate_desired_orientation(end_effector_position, target_position):
    direction_to_target = target_position - end_effector_position
    return direction_to_target / np.linalg.norm(direction_to_target)


xml_path = "franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

state_dim = model.nq + model.nv
goal_dim = 3
action_dim = model.nu
max_action = 1.0

sac = SAC(state_dim, goal_dim, action_dim, max_action)

num_episodes = 100
max_steps = 500
goal = np.array([0.5, 0.5, 0.5])
tolerance = 0.6
mean_d = []
median_d = []
min_d = []
tolerance_final = []
all_rewards = []


for episode in tqdm(range(num_episodes)):
    mujoco.mj_resetData(model, data)
    state = get_state(data)
    episode_reward = 0
    all_distances = []

    for step in range(max_steps):
        action = sac.select_action(state, goal)
        
        # Agregar ruido a las acciones para fomentar más exploración
        if 0 < episode < 10:
            noise_scale = 0.25
        if 0 < episode < 25:
            noise_scale = 0.2
        elif 25 <= episode < 50:
            noise_scale = 0.1
        elif 50 <= episode < 75:
            noise_scale = 0.05
        else:
            noise_scale = 0
        # noise_scale = max(0.01, 1 - ((episode / num_episodes) * 10))  # Reduce el ruido gradualmente
        noisy_action = action + noise_scale * np.random.randn(*action.shape)
        noisy_action = action + noise_scale * np.random.randn(*action.shape)
        apply_action = np.clip(noisy_action, -1, 1)  # Limitar la acción dentro de los límites

        data.ctrl[:] = apply_action
        mujoco.mj_step(model, data)

        next_state = get_state(data)
        reward, tolerance, distance_to_target = calculate_reward(data, goal, tolerance)
        all_distances.append(distance_to_target)
        done = step == max_steps - 1

        sac.add_to_buffer((state, apply_action, reward, next_state, done, goal))

        state = next_state
        episode_reward += reward

        if len(sac.replay_buffer.buffer) > 256:
            sac.train(batch_size=256)

        if done:
            break

    min_distance = min(all_distances)
    print(f"Episodio {episode + 1}, Recompensa Total: {episode_reward:.2f}")
    all_rewards.append(episode_reward)
    print(f"    Distancia más cercana: {min_distance:.2f}")
    min_d.append(min_distance)
    print(f"    Promedio distancias: {np.mean(all_distances):.2f}")
    mean_d.append(np.mean(all_distances))
    print(f"    Mediana distancia: {np.median(all_distances):.2f}")
    median_d.append(np.median(all_distances))
    print(f"    Tolerancia final: {tolerance:.2f}")
    tolerance_final.append(tolerance)

    # Guarda el modelo cada 50 episodios
    if (episode + 1) % 5 == 0:
        torch.save({
            "actor": sac.actor.state_dict(),
            "critic1": sac.critic1.state_dict(),
            "critic2": sac.critic2.state_dict(),
            "actor_optimizer": sac.actor_optimizer.state_dict(),
            "critic1_optimizer": sac.critic1_optimizer.state_dict(),
            "critic2_optimizer": sac.critic2_optimizer.state_dict(),
        }, f"ANAIS_sac_checkpoint_{episode + 1}.pth")
        fpickle.dump(f"listas_resultados/all_rewards_{episode + 1}.pickle", all_rewards)
        fpickle.dump(f"listas_resultados/min_distance_{episode + 1}.pickle", min_d)
        fpickle.dump(f"listas_resultados/mean_distance_{episode + 1}.pickle", mean_d)
        fpickle.dump(f"listas_resultados/median_distance_{episode + 1}.pickle", median_d)
        fpickle.dump(f"listas_resultados/tolerances_{episode + 1}.pickle", tolerance_final)

