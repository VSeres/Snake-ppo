import time
import os
import sys
from datetime import datetime
from typing import Callable
import pathlib
from threading import Thread, Lock
from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from snake import Snake2


class WindowsInhibitor:
    '''Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx'''
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        pass

    def inhibit(self):
        import ctypes
        print("Preventing Windows from going to sleep")
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS |
            WindowsInhibitor.ES_SYSTEM_REQUIRED)

    def uninhibit(self):
        import ctypes
        print("Allowing Windows to go to sleep")
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def evaluate(n_games: int, model: BaseAlgorithm, size: int) -> dict:
    """
    Ki értékeli a model

    :param n_games: hány kört játszon
    :param model: teszelendő model
    :param size: pálya mérete

    :return: statisztika
      'size': pálya mérete
      'avg_score': jutalmak átlaga
      'median': pontok átlaga
      'min_score': minimum pont
      'max_score': maximum pont
      'score_std': pontok szórása

    """
    lock = Lock()
    summation = {'scores': [], 'reward': []}

    def game():
        env = Snake2(size)
        reward_sum = 0
        obs = env.reset()
        done = False
        while not done:
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            reward_sum += reward
        lock.acquire(blocking=True)
        summation['scores'].append(info['score'])
        lock.release()

    processes = [Thread(target=game, daemon=True) for i in range(n_games)]
    [p.start() for p in processes]
    [p.join() for p in processes]
    statistics = {}
    statistics['size'] = f'{size}x{size}'
    statistics['avg_score'] = mean(summation['scores'])
    statistics['median'] = median(summation['scores'])
    statistics['min_score'] = min(summation['scores'])
    statistics['max_score'] = max(summation['scores'])
    statistics['score_std'] = std(summation['scores'])
    for k, v in statistics.items():
        print(f'{k}: {v}')
    return statistics


def make_env(i):
    """
    Létrehoza a DummyVecEnv()-hez szükséges wrapper fügvényeket

    :param i: pálya méretének megállapításához van használva
      minimum 0, maximum 9 lehet
    """
    if i < 1:  # 1
        size = 6
    elif i < 7:  # 6
        size = 9
    elif i < 15:  # 8
        size = 12
    else:  # 5
        size = 16

    def _f() -> Snake2:
        env = Snake2(size)
        check_env(env)
        return env
    return _f


def teach(n_env=40, n_epcoh=1, model='main16-16', shutdown=False, total_timesteps=1e8, learning_rate=2e-4, test_sizes=[6, 9, 12], new=False):
    """
    Modellt tanítja be

    :param n_env: hány környezeten tanuljon egyszere
    :param n_epoch: hányszor ismételje meg a tanulást
    :param model: model neve
    :param shutdown: számitógépet leállítja a végzet
    :param total_timesteps: összesen hány lépést tegyünk meg ez befolyásolja a tanulás időtartamát
    :param learning_rate: tanulási ráta, ez lehet egy függvény vagy egy constans
    :param test_sizes: mekkor pákákon tesztelje a modelt
    :param new: új modellt hozzon létre, felülirja ha már létezik
    """
    N_ENV = n_env
    N_EPOCH = n_epcoh
    PATH = str(pathlib.Path(__file__).parent.resolve())
    MODEL = PATH+'\\model\\'+model
    print(f'Model path: {MODEL}')
    env = [make_env(i % 20) for i in range(N_ENV)]
    env = DummyVecEnv(env)
    if new:
        model = PPO('MlpPolicy', env, verbose=1, batch_size=1024, n_epochs=15, policy_kwargs={
                    'net_arch': [dict(pi=[32, 16, 8], vf=[32, 16, 8])]},
                    learning_rate=learning_rate)
    else:
        model = PPO.load(MODEL, env=env)
        model.learning_rate = learning_rate
    osSleep = WindowsInhibitor()
    osSleep.inhibit()
    for n in range(N_EPOCH):
        start = time.time()
        model.learn(total_timesteps, reset_num_timesteps=True)
        end = time.time()
        total_time = end-start
        print(f'training: {total_time:.3f} s')
        model.save(MODEL)
        with open(PATH+"\\log.txt", "a") as log:
            log.write(f'--- {MODEL} ({n+1}) ---\n')
            log.write(
                f'net_arch: {model.policy_kwargs["net_arch"][0]["pi"]}\n')
            log.write(
                f'finished: {str(datetime.now())} ,  training time: {total_time:.3f} s\n')
            for size in test_sizes:
                statistics = evaluate(60, model, size)
                for k, v in statistics.items():
                    log.write(f'{k}: {v}\n')

    osSleep.uninhibit()
    if shutdown:
        os.system("shutdown.exe /s /t 5")
    env.close()
    sys.exit()


if __name__ == '__main__':
    teach(shutdown=False, total_timesteps=10e7, n_epcoh=1, n_env=60,
          learning_rate=linear_schedule(2e-4), model='32x16x8-3', new=False)
