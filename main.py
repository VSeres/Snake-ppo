from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median
from stable_baselines3 import PPO
from game import Snake
from Snake import Snake2
from stable_baselines3.common.env_checker import check_env
import time, os, sys
from datetime import datetime
from typing import Callable
from stable_baselines3.common.vec_env import DummyVecEnv
from threading import Thread, Lock

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
            WindowsInhibitor.ES_CONTINUOUS | \
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

def evaluate(n_games: int, model, *args, **kwargs) -> dict:
    lock = Lock()
    summation = {'rewards':[], 'scores':[]}
    def game():
        env = Snake2(*args, **kwargs)
        reward_sum = 0
        obs = env.reset()
        done = False
        while not done:
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            reward_sum += reward
        lock.acquire(blocking=True)
        summation['rewards'].append(reward_sum)
        summation['scores'].append(info['score'])
        lock.release()

    processes = [Thread(target=game) for i in range(n_games)]
    [p.start() for p in processes]
    [p.join()for p in processes]
    statistics = {}
    statistics['avg_reward'] = mean(summation['rewards'])
    statistics['avg_score'] = mean(summation['scores'])
    statistics['median'] = median(summation['scores'])
    statistics['min_score'] = min(summation['scores'])
    statistics['max_score'] = max(summation['scores'])
    statistics['score_std'] = std(summation['scores'])
    for k,v in statistics.items():
        print(f'{k}: {v}')
    return statistics

def makeEnv(i, *args, **kwargs):
    if i == 0:
        size = 6
    elif i == 1:
        size = 12
    else:
        size = 9
    def _f() -> Snake2:
        env = Snake2(width=size,height=size,*args, **kwargs)
        check_env(env)
        return env
    return _f

N_ENV = 12
N_EPOCH = 1
MODEL_NAME = './model/main'
if __name__ == '__main__':
    env = [makeEnv(i%4) for i in range(N_ENV)]
    #env = SubprocVecEnv(env)
    env = DummyVecEnv(env)
    #env = Snake2()
    # model = PPO('MlpPolicy', env, verbose=1, batch_size=512, policy_kwargs={'net_arch':[dict(pi=[64,32], vf=[64,32])] })
    osSleep = WindowsInhibitor()
    osSleep.inhibit()
    model = PPO.load(MODEL_NAME, env=env)
    for n in range(N_EPOCH):
        model.learning_rate = linear_schedule(0.00001)
        start = time.time()
        model.learn(total_timesteps=80000000, reset_num_timesteps=True)
        end = time.time()
        total_time = end-start
        print(f'training: {total_time:.3f} s')
        model.save(MODEL_NAME)
        model = PPO.load(MODEL_NAME, env=env)
        statistics = evaluate(60, model, 12, 12)
        with open("log.txt", "a") as log:
            log.write(f'--- {MODEL_NAME} ({n+1}/{N_EPOCH}) ---\n')
            log.write(f'number of environment: {N_ENV}\n')
            log.write(f'finished: {str(datetime.now())} ,  training time: {total_time:.3f} s\n')
            for k,v in statistics.items():
                log.write(f'{k}: {v}\n')
    osSleep.uninhibit()
    os.system("shutdown.exe /s /t 4")
    env.close()
    sys.exit()