import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import mujoco
import numpy as np
import funciones_pickle as fpickle
from tqdm import tqdm

#print(torch.__version__)
#print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state, goal):
        # Convertir a tensores si no lo son
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(next(self.parameters()).device)
        if not isinstance(goal, torch.Tensor):
            goal = torch.tensor(goal, dtype=torch.float32).to(next(self.parameters()).device)

        # Asegurarse dimensión batch
        # dimensión que agrupa varios ejemplos para procesarlos como un lote.
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Agregar dimensión batch
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)  # Agregar dimensión batch

        # Concatenar estado y meta
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
        z = normal.rsample()  # Reparameterization
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)  # Log probabilidad con tanh
        return action, log_prob.sum(1, keepdim=True)

# Define Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim + action_dim, 256)  # Estado + Meta + Acción
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, goal, action):
        x = torch.cat([state, goal, action], dim=1)  # Concatenar
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# replay buffer her
class HERReplayBuffer:
    def __init__(self, max_size, her_k=4):
        self.buffer = deque(maxlen=max_size)
        self.her_k = her_k  # Cuántas metas alternativas usar por transición.

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size, data):
        samples = random.sample(self.buffer, batch_size)
        augmented_samples = []

        for state, action, reward, next_state, done, goal in samples:
            # Generar metas alternativas basadas en `data.xpos[6]`
            her_goals = [data.xpos[6] for _ in range(self.her_k)]  # Usa la posición del efector final como nueva meta
            for new_goal in her_goals:
                # Calcula la nueva recompensa basada en la meta alternativa
                new_reward = -np.linalg.norm(np.array(new_goal) - np.array(goal))  # Penalización por distancia a la nueva meta
                augmented_samples.append((state, action, new_reward, next_state, done, new_goal))

        # Agregar las muestras originales
        augmented_samples.extend(samples)

        # Convertir a tensores
        states, actions, rewards, next_states, dones, goals = zip(*augmented_samples)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
                torch.tensor(goals, dtype=torch.float32))


# SAC Algorithm
class SAC:
    def __init__(self, state_dim, goal_dim, action_dim, max_action, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        # estados y metas concatenados
        self.actor = Actor(state_dim, goal_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, goal_dim, action_dim).to(device)
        self.target_critic1 = Critic(state_dim, goal_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, goal_dim, action_dim).to(device)

        # redes objetivo como clones de las críticas originales
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Replay Buffer con HER
        self.replay_buffer = HERReplayBuffer(max_size=1_000_000)

        # Hiperparámetros
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.max_action = max_action

    def select_action(self, state, goal):
        action, _ = self.actor.sample(state, goal)
        return action.cpu().detach().numpy().flatten()

    def train(self, batch_size=256):
        states, actions, rewards, next_states, dones, goals = self.replay_buffer.sample(batch_size, data)

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

        # Actualizar redes objetivo
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
    scaled_action = ctrl_min + (action + 1) * 0.5 * (ctrl_max - ctrl_min)  # Escala [-1, 1] al rango [ctrl_min, ctrl_max]
    data.ctrl[:] = scaled_action

def calculate_reward(data, target_position, all_distances):
    end_effector_position = data.xpos[6]
    distance_to_target = np.linalg.norm(end_effector_position - target_position)
    if len(all_distances) > 0:
        last_distance_to_target = all_distances[-1]
    else:
        last_distance_to_target = distance_to_target
    #all_distances.append(distance_to_target)
    distance_change = last_distance_to_target - distance_to_target
    reward = distance_change
    #print(f"distance: {distance_to_target}")
    #print(f"reward: {reward}")
    return reward, distance_to_target

def generate_random_goal(base_position=[0, 0, 0], radius=1):
    while True:
        random_offset = np.random.uniform(-radius, radius, size=3)
        random_goal = base_position + random_offset
        if np.linalg.norm(random_offset) <= radius:
            return random_goal


# Ruta al modelo XML
#xml_path = "franka_emika_panda/scene.xml"
xml_path = "franka_fr3_dual/scene.xml"

# Carga el modelo de MuJoCo
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Configuración SAC
state_dim = model.nq + model.nv  # Número de posiciones y velocidades
goal_dim = 3
action_dim = model.nu  # Número de controles
max_action = 1.0  # Acciones normalizadas [-1, 1]


sac = SAC(state_dim, goal_dim, action_dim, max_action)

num_episodes = 3000
max_steps = 50000  # Máximo de pasos por episodio
#goal = np.array([0.7, 0, 0.5])  # Meta fija

goal_tolerance = 0.40
base_goal_position = np.array([0, 0, 0])  # Posición base para los objetivos dinamicos
goal_radius = 1  # Radio máximo para los objetivos aleatorios

#for episode in range(num_episodes):
#    mujoco.mj_resetData(model, data)  # Reinicia la simulación
#    state = get_state(data)
#    episode_reward = 0
#    mean_d = []
#    median_d = []
#    min_d = []
#    #tolerance_final = []
#    all_rewards = []
#    all_distances = []
#
#    goal = generate_random_goal(base_goal_position)
#
#    step = 0
#    while True:  # Repetir indefinidamente hasta que se cumpla la condición
#        # Selecciona una acción
#        action = sac.select_action(state, goal)
#
#        # Aplica la acción y avanza la simulación
#        apply_action(data, action)
#        step_simulation(model, data)
#
#        # Extrae el nuevo estado, recompensa, y chequea si el episodio termina
#        next_state = get_state(data)
#        reward, distance_to_target = calculate_reward(data, goal, all_distances)
#        all_distances.append(distance_to_target)
#        print(goal, distance_to_target, step, reward, episode)
#        
#        # Verifica si se alcanzó el objetivo
#        done = distance_to_target <= goal_tolerance
#
#        # Agrega la transición al buffer
#        sac.add_to_buffer((state, action, reward, next_state, done, goal))
#
#        # Actualiza el estado
#        state = next_state
#        episode_reward += reward
#
#        # Entrena el modelo si hay suficientes datos
#        if len(sac.replay_buffer.buffer) > 256:
#            sac.train(batch_size=256)
#
#        if done or step >= 10000:
#            break  # Termina el bucle si se alcanza el objetivo
#
#        step += 1
#
#    # Registro del episodio
#    print(f"Episodio {episode + 1}, Recompensa Total: {episode_reward:.2f}")
#
#    min_distance = min(all_distances)
#    all_rewards.append(episode_reward)
#    min_d.append(min_distance)
#    mean_d.append(np.mean(all_distances))
#    median_d.append(np.median(all_distances))
#
#    # Guarda el modelo cada 50 episodios
#    if (episode + 1) % 5 == 0:
#        torch.save({
#        "actor": sac.actor.state_dict(),
#        "critic1": sac.critic1.state_dict(),
#        "critic2": sac.critic2.state_dict(),
#        "actor_optimizer": sac.actor_optimizer.state_dict(),
#        "critic1_optimizer": sac.critic1_optimizer.state_dict(),
#        "critic2_optimizer": sac.critic2_optimizer.state_dict(),
#        }, f"ANAIS_sac_checkpoint_{episode + 1}.pth")
#        fpickle.dump(f"listas_resultados/all_rewards_{episode + 1}.pickle", all_rewards)
#        fpickle.dump(f"listas_resultados/min_distance_{episode + 1}.pickle", min_d)
#        fpickle.dump(f"listas_resultados/mean_distance_{episode + 1}.pickle", mean_d)
#        fpickle.dump(f"listas_resultados/median_distance_{episode + 1}.pickle", median_d)



for episode in range(num_episodes):
    mujoco.mj_resetData(model, data)  # Reinicia la simulación
    state = get_state(data)
    episode_reward = 0
    all_rewards = []
    all_distances = []
    goal = generate_random_goal(base_goal_position, goal_radius)
    step = 0

    while step < max_steps:  # Detener cuando se alcance el máximo de pasos
        # Selecciona una acción
        action = sac.select_action(state, goal)

        # Aplica la acción y avanza la simulación
        apply_action(data, action)
        step_simulation(model, data)

        # Extrae el nuevo estado, recompensa, y chequea si el episodio termina
        next_state = get_state(data)
        reward, distance_to_target = calculate_reward(data, goal, all_distances)
        all_distances.append(distance_to_target)
        #print(goal, distance_to_target, step, reward)

        # Verifica si se alcanzó el objetivo
        done = distance_to_target <= goal_tolerance

        # Agrega la transición al buffer
        sac.add_to_buffer((state, action, reward, next_state, done, goal))

        # Actualiza el estado
        state = next_state
        episode_reward += reward

        # Incrementa los contadores
        step += 1

        # Entrenar el modelo si hay suficientes datos
        if len(sac.replay_buffer.buffer) > 256:
            sac.train(batch_size=256)

        # Guarda automáticamente cada 5000 pasos
        if step % 1000 == 0:
            print(f"Guardando en el paso global {step}")
            torch.save({
                "actor": sac.actor.state_dict(),
                "critic1": sac.critic1.state_dict(),
                "critic2": sac.critic2.state_dict(),
                "actor_optimizer": sac.actor_optimizer.state_dict(),
                "critic1_optimizer": sac.critic1_optimizer.state_dict(),
                "critic2_optimizer": sac.critic2_optimizer.state_dict(),
            }, f"ANAIS_sac_checkpoint_step_{step}.pth")
            fpickle.dump(f"listas_resultados/all_rewards_step_{step}.pickle", all_rewards)
            fpickle.dump(f"listas_resultados/all_distances_step_{step}.pickle", all_distances)

        if done:
            break  # Termina el bucle si se alcanza el objetivo
    
    if step >= max_steps:
        print(f"Se alcanzaron los {max_steps} pasos. Deteniendo el entrenamiento.")
        break