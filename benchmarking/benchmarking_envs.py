import numpy as np
import gym
from gym.envs.registration import register

from timer import Timer

register(
    id='CommonsGame_v0',
    entry_point='CommonsGame.envs:CommonsGame',
)


def main():

    numAgents = 11

    env = gym.make('CommonsGame_v0', numAgents=numAgents, visualRadius=4)
    env.reset()

    t = Timer()

    print("warmup")
    t.start()
    for _ in range(1000):
        nActions = np.random.randint(low=0, high=env.action_space.n, size=(numAgents,)).tolist()
        nObservations, nRewards, nDone, nInfo = env.step(nActions)

    t.stop()
    t.print_stats()

    t2 = Timer()
    t2.start()
    for ___ in range(10):
        for __ in range(1000):
            nActions = np.random.randint(low=0, high=env.action_space.n, size=(numAgents,)).tolist()
            nObservations, nRewards, nDone, nInfo = env.step(nActions)
        t2.update_episodes()

    t2.stop()
    t2.print_stats()


if __name__ == '__main__':
    main()