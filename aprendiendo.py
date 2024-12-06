from mujoco.glfw import glfw
import mujoco as mj
import numpy as np

from dataclasses import dataclass
import mediapy as media
from pathlib import Path
import enum
from tqdm import tqdm


model_dir = Path("franka_fr3")
model_xml = model_dir / "scene.xml"
t_sim = 10 # tiempo simulacion
# MuJoCo data structures: modelo, camara, opciones de visualizacion ---
model = mj.MjModel.from_xml_path(model_xml) # MuJoCo model
data = mj.MjData(model) # MuJoCo data
cam = mj.MjvCamera() # Abstract camara
opt = mj.MjvOption() # opciones de visualizacion


class Resolution(enum.Enum):
  SD = (480, 640)
  HD = (720, 1280)
  UHD = (2160, 3840)


def quartic(t: float) -> float:
  return 0 if abs(t) > 1 else (1 - t**2) ** 2


def blend_coef(t: float, duration: float, std: float) -> float:
  normalised_time = 2 * t / duration - 1
  return quartic(normalised_time / std)


def unit_smooth(normalised_time: float) -> float:
  return 1 - np.cos(normalised_time * 2 * np.pi)


def azimuth(
    time: float, duration: float, total_rotation: float, offset: float
) -> float:
  return offset + unit_smooth(time / duration) * total_rotation

res = Resolution.SD
fps = 60
duration = 10.0
ctrl_rate = 2
ctrl_std = 0.05
total_rot = 60
blend_std = .8
h, w = res.value


