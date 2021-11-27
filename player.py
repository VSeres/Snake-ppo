import pathlib
from stable_baselines3 import PPO
import numpy as np
import pygame

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

class HumanAgent:
    def __init__(self):
        self.previous_action = 3

    def step(self, *args):
        action = self.previous_action
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3
        self.previous_action = action
        return action