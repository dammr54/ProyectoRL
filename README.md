# ProyectoRL
 Este repositorio contiene el desarrollo de un modelo de aprendizaje por refuerzo (Reinforcement Learning, RL) diseñado para entrenar brazos robóticos Franka Emika Research 3 en un entorno de simulación.

 Se implementan los algoritmos SAC y HER para el entrenamiento.

 # Vista del entorno
 * El archivo main.py tiene una visualización del entorno sin control
 * El archivo probar1.py carga el modelo entrenado con **nuestra propia implementación**
 * El archivo probar2.py carga el modelo entrenado con la implementación de **stable-baselines3**

# Modelos
* franka_emika_panda contraresta la gravedad
* franka_fr3_dual no contraresta la gravedad, este se debe descomprimir desde el .zip
* Los archivos .pth tiene la información correspondiente a los pesos del actor para poder reproducir la simulación con la politica encontrada

# Train
* El archivo ```train.py``` contiene el código para entrenar el modelo con **nuestra propia implementación**.
* El archivo ```train_stable_baselines.py``` contiene el código para entrenar el modelo con la implementación de **stable-baselines3**. 
* Para replicar nuestro experimento <a target="_blank" href="https://colab.research.google.com/github/dammr54/ProyectoRL/blob/main/run%20train.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Data
* La data se encuentra comprimida en los archivos pickle en la carpeta listas_resultados


# Otros
* funciones_pickle.py contiene funciones utiles para escritura y lectura de datos

