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

import argparse
import time
import yaml

import env

from action import Action
from coord import Coord
from random_agent import RandomAgent
from mdp_agent import MdpAgent
from sarsa_agent import SarsaAgent
from q_agent import QAgent


def draw_header():
    print("\033c", end="")
    print(
        """
 .d8888b.          d8b      888 888       888                  888      888
d88P  Y88b         Y8P      888 888   o   888                  888      888
888    888                  888 888  d8b  888                  888      888
888        888d888 888  .d88888 888 d888b 888  .d88b.  888d888 888  .d88888
888  88888 888P"   888 d88" 888 888d88888b888 d88""88b 888P"   888 d88" 888
888    888 888     888 888  888 88888P Y88888 888  888 888     888 888  888
Y88b  d88P 888     888 Y88b 888 8888P   Y8888 Y88..88P 888     888 Y88b 888
 "Y8888P88 888     888  "Y88888 888P     Y888  "Y88P"  888     888  "Y88888

    """
    )


if __name__ == "__main__":
    draw_header()

    parser = argparse.ArgumentParser(
        description="Gridworld with different RL algorithms"
    )
    parser.add_argument("cfg", type=str, help="Path to .yaml configuration file")
    args = parser.parse_args()

    with open(args.cfg, "r") as file:
        cfg = yaml.safe_load(file)

    env = env.GridWorld(cfg)
    print(env.world.grid[11][11])

    atype = cfg["agent"]["type"].lower()
    if atype in ["random", "rand"]:
        agent = RandomAgent(env.action_space)
    elif atype == "mdp":
        agent = MdpAgent(env.action_space)
    elif atype == "sarsa":
        agent = SarsaAgent(env.action_space)
    elif atype == "q":
        agent = QAgent(env.action_space)
    else:
        raise f"Unknown type of agent {atype}.."

    state = env.gen_obs()
    env.render()

    stop = False
    n_steps = 1
    actions = ""

    while not stop:
        while True:
            running_mode = input(
                "🚀 #Steps to run? [\033[1;35mint(default: 1)\033[0m|fire|{quit|q}] "
            ).strip()

            if running_mode in ["quit", "q"]:
                stop = True
                break
            elif running_mode == "fire":
                n_steps = cfg["max_steps"]
                break
            else:
                try:
                    if running_mode == "":
                        break
                    else:
                        n_steps = int(running_mode)
                        break
                except ValueError:
                    print(f"\033[0;31mUnknown option..\033[0m")
                    time.sleep(0.5)

        if not stop:
            for _ in range(n_steps):
                curr_state = env.world.agent_pos

                action = agent(state)
                actions += f" {env.idx_to_action[action]}"

                state, reward, done, _ = env.step(action)

                if done == True:
                    stop = True
                    break

            draw_header()
            env.render()
            print(
                f"\033[1;34mLast reward:\033[0m {reward} [ from: {curr_state}, to: {env.world.agent_pos} ]"
            )
            print("\033[1;34mHistory    :\033[0m", end="")
            print(actions)

            n_steps = 1

    # for _ in range(cfg['max_steps']):
    #     action = agent(state)
    #     print(env.actions_idx[action])
    #     new_state, reward, done, _ = env.step(action)

    #     # Update value function / policy
    #     print(env.dist[state][action])

    #     # New state is state
    #     state = new_state

    #     env.render()
    #     if done == True:
    #         break

    #     time.sleep(1.0)
