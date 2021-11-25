import pathlib
from stable_baselines3 import PPO
import numpy as np


class PPOAgent:
    def __init__(self, model_name: str = 'main16-16'):
        self.model_route = str(pathlib.Path(
            __file__).parent.resolve())+'/model/'+model_name
        self.load_model()

    def load_model(self) -> None:
        self.model = PPO.load(self.model_route)

    def step(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs)
        return int(action)
