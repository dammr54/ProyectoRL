import mujoco # MuJoCo (Multi-Joint dynamics with Contact - Dinámica multiarticular con contacto)
from mujoco.glfw import glfw # GLFW (Graphics Library Framework - Marco de biblioteca de gráficos)
# manejo de ventanas, entrada y contexto OpenGL, que se utiliza frecuentemente en simulaciones que requieren renderizado en tiempo real
# Crear y manejar ventanas, procesar eventos de entrada y tiene soporte multiplataforma
# GLFW proporciona el contexto OpenGL -> depuración visual
# Renderizar: Generar una representación visual a partir de datos o información
import OpenGL.GL as gl # API estándar ampliamente utilizada para el desarrollo de gráficos 2D y 3D, permitiendo la interacción directa con hardware gráfico
# permiten enviar datos desde la CPU hacia la GPU para el procesamiento y renderizado.
# Controlar el pipeline gráfico -> transformación, proyección, rasterización y sombreado
# control básico y avanzado del pipeline gráfico, permitiéndote construir gráficos interactivos y de alto rendimiento
import time
import numpy as np # manejo de la data
import gym
from gym import spaces

# ---------------- Definición del modelo --------------------------
#xml_path = "franka_fr3/scene.xml"  # Ruta archivo XML con el modelo sin grippers
#xml_path = "franka_emika_panda/scene.xml"  # Ruta archivo XML con el modelo con grippers
#xml_path = "car1.xml"
xml_path = "franka_fr3_dual/scene.xml"
# cambios

# --- MuJoCo data structures: modelo, cámara, opciones de visualización ---
model = mujoco.MjModel.from_xml_path(xml_path)  # MuJoCo model -> fisica, geometria y cinematica
data = mujoco.MjData(model)  # MuJoCo data -> información dinámica y temporal
cam = mujoco.MjvCamera()  # Abstract camera
opt = mujoco.MjvOption()  # personalizar renderizacion

#print(dir(data))
#help(data)
data.ctrl[:] = [10., 0., 0., 0., 0., 0., 0., 0., 0.]
#print(model.actuator_ctrlrange)
#data.qpos[6] = 2
print(data.ctrl)

#print(data.ctrl[:])

# Configuración inicial de la cámara -> punto a enfocar
cam.azimuth = 90 # rotacion
cam.elevation = -8 # elevacion eje Z
cam.distance = 4 # distancia de camara al objeto
cam.lookat = np.array([0.0, 0.0, 1]) # punto hacia donde está enfocada la cámara [x, y, z]

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
    # La sincronización vertical (v-sync) asegura que los cuadros renderizados por tu programa se sincronizan con la frecuencia de actualización del monitor
    # evitar problemas graficos

    # Registrar los callbacks
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, mouse_scroll)
    glfw.set_cursor_pos_callback(window, mouse_move)

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
    #else:
    #    print('Mouse button released at: ', glfw.get_cursor_pos(window))

# Callback de desplazamiento del mouse (zoom)
def mouse_scroll(window, xoffset, yoffset):
    action = mujoco.mjtMouse.mjMOUSE_ZOOM
    mujoco.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)

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

    # shift key state
    PRESS_LEFT_SHIFT  = glfw.get_key(window, glfw.KEY_LEFT_SHIFT)  == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # Determinar la acción basada en el botón del mouse
    if mouse_button_right: # si el botón derecho está presionado
        # Si Shift está presionado, mover horizontalmente; de lo contrario, mover verticalmente
        if mod_shift:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H  # Movimiento horizontal
        else:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V  # Movimiento vertical

        mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)  # Mover la cámara
    
    if mouse_button_left:  # Rotación de la cámara con el botón izquierdo
        # Si shift está presionado, rotar horizontalmente; de lo contrario, rotar verticalmente
        if mod_shift:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H  # Rotación horizontal
        else:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V  # Rotación vertical

        mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)  # Rotar la cámara

# --- Función principal ---
def main():
    window = glfw_init()

    # Crear el contexto de visualización (renderizado )
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    # Bucle de visualización GLFW
    while not glfw.window_should_close(window):
        # Obtener y ejecutar eventos
        glfw.poll_events()

        # Actualizar la simulación
        mujoco.mj_step(model, data)

        # Obtener los datos del sensor (opcional, pero útil)
        #print('Posición (qpos):', data.qpos[:3])
        #print('Velocidad (qvel):', data.qvel[:3])

        # Obtener el tamaño del framebuffer -> renderizacion
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        # Actualizar la escena y renderizar
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)

        # Intercambiar los buffers (front y back) -> donde se renderizó la escena con lo que actualmente está visible en la ventana
        glfw.swap_buffers(window)

    glfw.terminate()

# ejecutar función principal
if __name__ == '__main__':
    main()