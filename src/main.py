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
    max_step = 1000
    env = env.GridWorld(6)
    
    agent = agent.RandomAgent(env.action_space)
    #agent = agent.StairsAgent(env.action_space)
    
    state = env.reset()
    for _ in range(max_step):
        action = agent(state)
        new_state, reward, done, _ = env.step(action)

        # Update value function / policy

        # New state is state
        state = new_state

        env.render()
        if done == True: 
            break
        time.sleep(1.0)
