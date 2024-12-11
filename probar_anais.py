import mujoco
import numpy as np
import torch
import torch.nn as nn
import os
from PIL import Image

# --- Configuración del entorno MuJoCo ---
xml_path = "franka_emika_panda/scene.xml"  # Ruta del archivo XML
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()

# Configuración inicial de la cámara
cam.azimuth = 90
cam.elevation = -8
cam.distance = 4
cam.lookat = np.array([0.0, 0.0, 1])
mujoco.mjv_defaultOption(opt)
scene = mujoco.MjvScene(model, maxgeom=10000)
mujoco.mj_forward(model, data)

# --- Función para guardar imágenes ---
def render_and_save_frame(model, data, scene, cam, filename="frame.png"):
    viewport = mujoco.MjrRect(0, 0, 640, 480)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    rgb_array = np.zeros((viewport.height, viewport.width, 3), dtype=np.uint8)
    depth_array = np.zeros((viewport.height, viewport.width), dtype=np.float32)
    mujoco.mjr_render(viewport, scene, context)
    mujoco.mjr_readPixels(rgb_array, depth_array, viewport)
    img = Image.fromarray(rgb_array)
    img.save(filename)

context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# --- Definición del modelo Actor ---
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return mean, std

# Dimensiones de estado y acción
state_dim = 34
action_dim = 8

# Cargar modelo del actor
actor = ActorNetwork(state_dim, action_dim)
model_path = "ANAIS_sac_checkpoint_100.pth"
checkpoint = torch.load(model_path)
actor.load_state_dict(checkpoint["actor"])
actor.eval()
print("Modelo Actor cargado correctamente.")

# --- Configuración de simulación ---
target_positions = [
    np.array([-0.7, 0, 0.5]),
    np.array([-0.5, -0.5, 0.2]),
    np.array([0.0, 0.5, 0.5]),
    np.array([0.5, 0.0, 0.3])
]
success_threshold = 0.05

def calculate_reward(data, target_position):
    current_position = data.xpos[6]
    distance = np.linalg.norm(current_position - target_position)
    return -distance

# --- Bucle de simulación ---
def main():
    cumulative_reward = 0
    frame_counter = 0

    os.makedirs("simulation_frames", exist_ok=True)

    for target_position in target_positions:
        print(f"Intentando alcanzar el objetivo: {target_position}")
        steps_to_reach = 0

        while True:
            # Renderizar y guardar el frame
            render_and_save_frame(model, data, scene, cam, filename=f"simulation_frames/frame_{frame_counter:04d}.png")
            frame_counter += 1

            # Obtener el estado actual
            state = np.concatenate([data.qpos, data.qvel, target_position])
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, std = actor(state_tensor)
            action = mean.detach().numpy().squeeze()

            # Aplicar la acción al entorno
            data.ctrl[:] = action
            mujoco.mj_step(model, data)

            # Calcular recompensa
            reward = calculate_reward(data, target_position)
            cumulative_reward += reward
            steps_to_reach += 1

            # Verificar si el objetivo se alcanzó
            if np.linalg.norm(data.xpos[6] - target_position) < success_threshold:
                print(f"Objetivo {target_position} alcanzado en {steps_to_reach} pasos.")
                break

    print(f"Recompensa acumulada total: {cumulative_reward}")

if __name__ == "__main__":
    main()
