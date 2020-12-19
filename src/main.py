#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "0.1.0"
__maintainer__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]

import time
import yaml
import agent
import env


if __name__ == "__main__":
    #agent = agent.RandomAgent()

    env = env.GridWorld(6, 10)
    env.reset()
    obs, reward, done, _ = env.step(env.action_space.sample())
    print(obs)
    print(reward)
    env.render()

    # for _ in range(100):
    #     env.step(env.action_space.sample()) # take a random action
    #     time.sleep(0.5)
    #     env.render()

    # for _ in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample()) # take a random action
    # env.close()
