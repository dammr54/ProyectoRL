import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gym
import mujoco
import numpy as np

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

        # Asegurarse de que state y goal tengan una dimensión batch
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
        z = normal.rsample()  # Reparameterization trick
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
        """
        Buffer con soporte para Hindsight Experience Replay.
        Args:
            max_size (int): Tamaño máximo del buffer.
            her_k (int): Número de objetivos alternativos generados por transición.
        """
        self.buffer = deque(maxlen=max_size)
        self.her_k = her_k  # Cuántas metas alternativas usar por transición.

    def add(self, transition):
        """Agrega una transición al buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size):
        """Toma muestras aleatorias y aplica HER."""
        samples = random.sample(self.buffer, batch_size)
        augmented_samples = []

        for state, action, reward, next_state, done, goal in samples:
            # Meta alternativa: usa el estado final como la nueva meta
            # her_goals = [next_state[:3] for _ in range(self.her_k)]  # Usa 3D posición del brazo como nueva meta
            her_goals = [np.random.uniform(low=-1, high=1, size=3) for _ in range(self.her_k)]
            for new_goal in her_goals:
                # Calcula nueva recompensa con la meta redefinida
                new_reward = -np.linalg.norm(next_state[:3] - new_goal)  # Penalización por distancia
                augmented_samples.append((state, action, new_reward, next_state, done, new_goal))

        # Devuelve las muestras originales + muestras augmentadas
        augmented_samples.extend(samples)
        states, actions, rewards, next_states, dones, goals = zip(*augmented_samples)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
                torch.tensor(goals, dtype=torch.float32))


# SAC Algorithm
class SAC:
    def __init__(self, state_dim, goal_dim, action_dim, max_action, lr=3e-4, gamma=0.95, tau=0.005, alpha=0.1):
        """
        Inicializa el modelo SAC con HER.

        Args:
            state_dim (int): Dimensión del estado.
            goal_dim (int): Dimensión de la meta (goal).
            action_dim (int): Dimensión de las acciones.
            max_action (float): Valor máximo permitido para las acciones (normalizado).
            lr (float): Tasa de aprendizaje.
            gamma (float): Factor de descuento.
            tau (float): Factor de interpolación suave para las redes objetivo.
            alpha (float): Parámetro de entropía para SAC.
        """
        # Ajustar redes para incluir estados y metas concatenados
        self.actor = Actor(state_dim, goal_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, goal_dim, action_dim).to(device)
        self.target_critic1 = Critic(state_dim, goal_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, goal_dim, action_dim).to(device)

        # Inicializar las redes objetivo como clones de las críticas originales
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
        # Convertir state y goal a tensores si no lo son
        #state = torch.tensor(state, dtype=torch.float32).to(device) if not isinstance(state, torch.Tensor) else state
        #goal = torch.tensor(goal, dtype=torch.float32).to(device) if not isinstance(goal, torch.Tensor) else goal

        # Concatenar estado y meta
        #state_goal = torch.cat([state, goal], dim=0).unsqueeze(0).to(device)  # Agregar dimensión para batch
        action, _ = self.actor.sample(state, goal)
        return action.cpu().detach().numpy().flatten()

    def train(self, batch_size=256):
        states, actions, rewards, next_states, dones, goals = self.replay_buffer.sample(batch_size)

        states, actions, rewards, next_states, dones, goals = (
            states.to(device), actions.to(device), rewards.to(device),
            next_states.to(device), dones.to(device), goals.to(device)
        )

        #state_goal = torch.cat([states, goals], dim=1)
        #next_state_goal = torch.cat([next_states, goals], dim=1)

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
    """Obtiene el estado actual del sistema."""
    # Combina posiciones y velocidades de las articulaciones
    state = np.concatenate([data.qpos, data.qvel])
    return state

def step_simulation(model, data):
    """Avanza la simulación en MuJoCo."""
    mujoco.mj_step(model, data)

def apply_action(data, action):
    """Aplica la acción al sistema."""
    action = np.clip(action, -1, 1)
    ctrl_min = model.actuator_ctrlrange[:, 0]
    ctrl_max = model.actuator_ctrlrange[:, 1]
    scaled_action = ctrl_min + (action + 1) * 0.5 * (ctrl_max - ctrl_min)  # Escala [-1, 1] al rango [ctrl_min, ctrl_max]
    data.ctrl[:] = scaled_action


def calculate_reward(data, target_position, target_orientation=None, tolerance=0.1):
    """
    Calcula la recompensa para el brazo robótico basado en la distancia al objetivo,
    orientación deseada y esfuerzo aplicado.

    Args:
        data (mujoco.MjData): Datos dinámicos de MuJoCo.
        target_position (np.array): Coordenadas 3D de la posición objetivo (x, y, z).
        target_orientation (np.array, optional): Orientación objetivo (cuaternión o matriz de rotación).
        tolerance (float): Distancia tolerada para considerar que el objetivo fue alcanzado.

    Returns:
        float: Valor de la recompensa.
    """
    end_effector_id = 6

    # Obtener la posición actual del end-effector
    end_effector_position = data.xpos[end_effector_id]

    # Calcular distancia al objetivo
    distance_to_target = np.linalg.norm(end_effector_position - target_position)

    # Penalización por distancia al objetivo
    # reward = -distance_to_target*100
    reward = -distance_to_target
    if distance_to_target < tolerance:
        reward += 1.0
    reward /= 10  # Escalado


    # Bonificación por éxito si está dentro de la tolerancia
    # if distance_to_target < tolerance:
    #     reward += 100.0  # Bonificación fija por alcanzar el objetivo

    # Penalización opcional por desalineación de orientación
    if target_orientation is not None:
        # Orientación actual del end-effector (cuaternión)
        current_orientation = data.xquat[end_effector_id]
        # Diferencia de orientación (puedes ajustar según la métrica que desees usar)
        orientation_diff = np.linalg.norm(current_orientation - target_orientation)
        reward -= 0.1 * orientation_diff  # Penalización leve por desalineación

    # Penalización por esfuerzo aplicado
    #control_effort = np.sum(np.square(data.ctrl))  # Esfuerzo total en los actuadores
    #reward -= 0.01 * control_effort

    return reward


# Ruta al modelo XML
xml_path = "franka_emika_panda/scene.xml"

# Carga el modelo de MuJoCo
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Configuración SAC
state_dim = model.nq + model.nv  # Número de posiciones y velocidades
goal_dim = 3
action_dim = model.nu  # Número de controles
max_action = 1.0  # Acciones normalizadas [-1, 1]


sac = SAC(state_dim, goal_dim, action_dim, max_action)

num_episodes = 120
max_steps = 100  # Máximo de pasos por episodio
goal = np.array([-0.7, 0, 0.5])  # Meta fija, cambiar si es dinámico

for episode in range(num_episodes):
    mujoco.mj_resetData(model, data)  # Reinicia la simulación
    state = get_state(data)
    episode_reward = 0

    for step in range(max_steps):
        # Selecciona una acción
        # action = sac.select_action(state, goal)
        action = sac.select_action(state, goal)
        if episode < 10:  # Agregar ruido durante los primeros episodios
            action += np.random.normal(0, 0.1, size=action.shape)


        # Aplica la acción y avanza la simulación
        apply_action(data, action)
        step_simulation(model, data)

        # Extrae el nuevo estado, recompensa, y chequea si el episodio termina
        next_state = get_state(data)
        reward = calculate_reward(data, goal)
        done = step == max_steps - 1  # Termina después de un número fijo de pasos

        # Agrega la transición al buffer
        sac.add_to_buffer((state, action, reward, next_state, done, goal))

        # Actualiza el estado
        state = next_state
        episode_reward += reward

        # Entrena el modelo si hay suficientes datos
        if len(sac.replay_buffer.buffer) > 256:
            sac.train(batch_size=256)

        if done:
            break

    print(f"Episodio {episode + 1}, Recompensa Total: {episode_reward:.2f}")

    # Guarda el modelo cada 50 episodios
    if (episode + 1) % 30 == 0:
        torch.save({
        "actor": sac.actor.state_dict(),
        "critic1": sac.critic1.state_dict(),
        "critic2": sac.critic2.state_dict(),
        "actor_optimizer": sac.actor_optimizer.state_dict(),
        "critic1_optimizer": sac.critic1_optimizer.state_dict(),
        "critic2_optimizer": sac.critic2_optimizer.state_dict(),
    }, f"sac_checkpoint_{episode + 1}.pth")