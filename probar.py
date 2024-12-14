import mujoco
from mujoco.glfw import glfw
import OpenGL.GL as gl
import numpy as np
import torch
import torch.nn as nn

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

# Variables para interacción con GLFW
mouse_lastx, mouse_lasty = 0, 0
mouse_button_left, mouse_button_middle, mouse_button_right = False, False, False

# --- Funciones de interacción con GLFW ---
def mouse_button(window, button, act, mods):
    global mouse_button_left, mouse_button_middle, mouse_button_right
    mouse_button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    mouse_button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    mouse_button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

def mouse_scroll(window, xoffset, yoffset):
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)

def mouse_move(window, xpos, ypos):
    global mouse_lastx, mouse_lasty
    dx, dy = xpos - mouse_lastx, ypos - mouse_lasty
    mouse_lastx, mouse_lasty = xpos, ypos
    if not (mouse_button_left or mouse_button_middle or mouse_button_right):
        return
    width, height = glfw.get_window_size(window)
    if mouse_button_right:
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_V, dx / height, dy / height, scene, cam)
    if mouse_button_left:
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx / height, dy / height, scene, cam)

def glfw_init():
    width, height = 1280, 720
    if not glfw.init():
        print("No se pudo inicializar GLFW")
        exit(1)
    window = glfw.create_window(width, height, "Visualización MuJoCo", None, None)
    if not window:
        glfw.terminate()
        print("No se pudo inicializar la ventana GLFW")
        exit(1)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, mouse_scroll)
    glfw.set_cursor_pos_callback(window, mouse_move)
    return window

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)  # Si el actor tiene log_std (común en SAC)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))  # Acción promedio
        log_std = self.log_std(x)      # Salida de log_std
        std = torch.exp(log_std)       # Convertir log_std a std
        return mean, std

# Dimensiones de estado y acción (ajustar según el entorno)
state_dim = 34  # Ejemplo: concatenación de qpos y qvel
action_dim = 8  # Número de actuadores controlados

# Crear y cargar el modelo del actor
actor = ActorNetwork(state_dim, action_dim)
#model_path = "modelos_entrenados/sac_checkpoint_100.pth"
#model_path = "sac_checkpoint_1600.pth"
model_path = "ANAIS_sac_checkpoint_100.pth"
#model_path = "policy.pth"
checkpoint = torch.load(model_path)
#checkpoint = dict(checkpoint)
#print("checkpoint", checkpoint)
#print("checkpoint[\"actor\"]", checkpoint["actor"])
actor.load_state_dict(checkpoint["actor"])
#actor.load_state_dict(checkpoint["actor.latent_pi.0.weight"])
actor.eval()  # Modo evaluación
print("Modelo Actor cargado correctamente.")

# --- Configuración de simulación ---
target_positions = [
    np.array([-0.7, 0, 0.5]),
    np.array([-0.5, -0.5, 0.2]),
    np.array([0.0, 0.5, 0.5]),
    np.array([0.5, 0.0, 0.3])
]  # Puedes añadir más objetivos según sea necesario
success_threshold = 0.05


def calculate_reward(data, target_position):
    current_position = data.qpos[6]  # Posición actual del actuador
    distance = np.linalg.norm(current_position - target_position)
    return -distance

# --- Bucle de simulación ---
def main():
    window = glfw_init()
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    cumulative_reward = 0
    for target_position in target_positions:
        print(f"Intentando alcanzar el objetivo: {target_position}")
        steps_to_reach = 0  # Contador para el número de pasos por objetivo

        while True:
            if glfw.window_should_close(window):
                glfw.terminate()
                print("Ventana cerrada. Finalizando simulación.")
                return

            glfw.poll_events()

            # Obtener el estado actual
            state = np.concatenate([data.qpos, data.qvel, target_position])
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, std = actor(state_tensor)
            action = mean.detach().numpy().squeeze()

            # Aplicar la acción al entorno
            data.ctrl[:] = action
            mujoco.mj_step(model, data)

            # Calcular recompensa y verificar si el objetivo se alcanzó
            reward = calculate_reward(data, target_position)
            cumulative_reward += reward
            steps_to_reach += 1

            print(data.xpos[6])

            if np.linalg.norm(data.xpos[6] - target_position) < success_threshold:
                print(f"Objetivo {target_position} alcanzado en {steps_to_reach} pasos.")
                break  # Pasar al siguiente objetivo

            # Renderizado
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)

    print(f"Recompensa acumulada total: {cumulative_reward}")
    glfw.terminate()

if __name__ == '__main__':
    main()