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
    with open("conf.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    max_step = 1000
    env = env.GridWorld(cfg)

    agent = agent.RandomAgent(env.action_space)
    # agent = agent.StairsAgent(env.action_space)

    state = env.reset()
    for _ in range(max_step):
        action = agent(state)
        print(env.actions_idx[action])
        new_state, reward, done, _ = env.step(action)

        # Update value function / policy
        print(env.dist[state][action])

        # New state is state
        state = new_state

        env.render()
        if done == True:
            break
        time.sleep(1.0)
