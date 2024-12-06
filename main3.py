import mujoco
from mujoco.glfw import glfw
import OpenGL.GL as gl
import time
import numpy as np

# --- Definición de un modelo ---
xml_path = "franka_fr3/scene.xml"  # Ruta a tu archivo XML de MuJoCo

# --- MuJoCo data structures: modelo, cámara, opciones de visualización ---
model = mujoco.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mujoco.MjData(model)  # MuJoCo data
cam = mujoco.MjvCamera()  # Abstract camera
opt = mujoco.MjvOption()  # Visualization options

# Configuración inicial de la cámara
cam.azimuth = 89.608063
cam.elevation = -11.588379
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 1.5])

# Inicialización de estructuras de visualización
mujoco.mjv_defaultOption(opt)
scene = mujoco.MjvScene(model, maxgeom=10000)

# Actualización cinemática
mujoco.mj_forward(model, data)

# --- Inicialización del motor gráfico OpenGL vía GLFW ---
def glfw_init():
    width, height = 1280, 720
    window_name = 'Visualización de MuJoCo'

    if not glfw.init():
        print("No se pudo inicializar el contexto OpenGL")
        exit(1)

    # Crear la ventana y su contexto OpenGL
    window = glfw.create_window(width, height, window_name, None, None)
    if not window:
        glfw.terminate()
        print("No se pudo inicializar la ventana GLFW")
        exit(1)

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # Activar v-sync

    return window

# Variables para el movimiento del mouse
mouse_lastx, mouse_lasty = 0, 0
mouse_button_left, mouse_button_middle, mouse_button_right = False, False, False

# Callback de los botones del mouse
def mouse_button(window, button, act, mods):
    global mouse_button_left, mouse_button_middle, mouse_button_right
    
    # Actualizar el estado de los botones
    mouse_button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    mouse_button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    mouse_button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # Actualizar posición del mouse cuando el botón es presionado o liberado
    if mouse_button_left or mouse_button_middle or mouse_button_right:
        print('Mouse button pressed at:  ', glfw.get_cursor_pos(window))
    else:
        print('Mouse button released at: ', glfw.get_cursor_pos(window))

# Callback de desplazamiento del mouse (zoom)
def mouse_scroll(window, xoffset, yoffset):
    action = mujoco.mjtMouse.mjMOUSE_ZOOM
    mujoco.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)
    # Hacer zoom con la rueda del mouse
    # print('Mouse scroll')

# Callback de movimiento del mouse
def mouse_move(window, xpos, ypos):
    global mouse_lastx, mouse_lasty
    global mouse_button_left, mouse_button_middle, mouse_button_right
    
    # Calcular desplazamiento del mouse
    dx = xpos - mouse_lastx
    dy = ypos - mouse_lasty
    mouse_lastx = xpos
    mouse_lasty = ypos

    # No hacer nada si no hay botones presionados
    if not (mouse_button_left or mouse_button_middle or mouse_button_right):
        return

    # Obtener tamaño actual de la ventana
    width, height = glfw.get_window_size(window)

    # Determinar la acción basada en el botón del mouse
    if mouse_button_right:  # Solo si el botón derecho está presionado
        # Si Shift está presionado, mover horizontalmente; de lo contrario, mover verticalmente
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H  # Movimiento horizontal
        else:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V  # Movimiento vertical

        mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)  # Mover la cámara
    
    if mouse_button_left:  # Rotación de la cámara con el botón izquierdo
        # Si Shift está presionado, rotar horizontalmente; de lo contrario, rotar verticalmente
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H  # Rotación horizontal
        else:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V  # Rotación vertical

        mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)  # Rotar la cámara

# --- Función principal ---
def main():
    window = glfw_init()

    # Registrar los callbacks
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, mouse_scroll)
    glfw.set_cursor_pos_callback(window, mouse_move)

    # Crear el contexto de visualización
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    # Bucle de visualización GLFW
    while not glfw.window_should_close(window):
        # Obtener y ejecutar eventos
        glfw.poll_events()

        # Actualizar la simulación
        mujoco.mj_step(model, data)

        # Obtener los datos del sensor (opcional, pero útil)
        print('Posición (qpos):', data.qpos[:3])
        print('Velocidad (qvel):', data.qvel[:3])

        # Obtener el tamaño del framebuffer
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        # Actualizar la escena y renderizar
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)

        # Intercambiar los buffers (front y back)
        glfw.swap_buffers(window)

    glfw.terminate()

# Ejecutar la función principal
if __name__ == '__main__':
    main()
