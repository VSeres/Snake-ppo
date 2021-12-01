import pathlib
from stable_baselines3 import PPO
import numpy as np
import pygame


class Agent:
    def step(self, obs: np.ndarray) -> int:
        raise NotImplementedError


class PPOAgent(Agent):
    """Ez az agent a stable baselines 3 ban betanított PPO modelt használja

    :vari model_route: abszolút útvonal a modellhez
    """

    def __init__(self, model_name: str = 'main16-16'):
        """
        a modell a ./model mappábol lesz betöltve

        :param model_name: a használandó model neve
        """
        self.model_route = str(pathlib.Path(
            __file__).parent.resolve())+'/model/'+model_name
        self.load_model()

    def load_model(self) -> None:
        """A model betöltéséhez használt fügvény.
        Ez megtörténik automatikusan az agent létrehozásakor"""
        self.model = PPO.load(self.model_route)

    def step(self, obs: np.ndarray) -> int:
        """
        Agent lépése
        :param obs: A környezet állapota
        :return: Az agent lépése
        """
        action, _ = self.model.predict(obs)
        return int(action)


class HumanAgent(Agent):
    """Az emberi játékost megvalósítása"""

    def __init__(self):
        self.previous_action = 3

    def step(self, *unused):
        """
        Agent lépése

        :return: Az agent lépése
        """
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
