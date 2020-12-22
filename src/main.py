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

GRID_CFG = ""
AGENT_CFG = ""


def draw_header():
    print("\033c", end="")
    print(
        f"""\
    .d8888b.          d8b      888 888       888                  888      888
    d88P  Y88b         Y8P      888 888   o   888                  888      888
    888    888                  888 888  d8b  888                  888      888
    888        888d888 888  .d88888 888 d888b 888  .d88b.  888d888 888  .d88888
    888  88888 888P"   888 d88" 888 888d88888b888 d88""88b 888P"   888 d88" 888
    888    888 888     888 888  888 88888P Y88888 888  888 888     888 888  888
    Y88b  d88P 888     888 Y88b 888 8888P   Y8888 Y88..88P 888     888 Y88b 888
    "Y8888P88 888     888  "Y88888 888P     Y888  "Y88P"  888     888  "Y88888
    {AGENT_CFG} {GRID_CFG}
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
    if "traps" in cfg["world"] and cfg["world"]["traps"]["type"] == "random":
        GRID_CFG = f'[ trap density: {cfg["world"]["traps"]["dist"]["trap"]} ]'

    atype = cfg["agent"]["type"].lower()
    if atype in ["random", "rand"]:
        AGENT_CFG = '[ agent: "random" ]'
        agent = RandomAgent(env.actions.to_indices(), **cfg["world"]["start"])
    elif atype == "mdp":
        AGENT_CFG = f'[ agent: "mdp", discount: {cfg["agent"]["discount"]}, obey: {cfg["agent"]["discount"]} ]'
        agent = MdpAgent(
            env.actions.to_indices(),
            cfg["world"]["start"]["x"],
            cfg["world"]["start"]["y"],
            env.transitions,
            cfg["agent"]["discount"],
            cfg["agent"]["obey_factor"],
        )
    # elif atype == "sarsa":
    #     agent = SarsaAgent(env.action_space)
    # elif atype == "q":
    #     agent = QAgent(env.action_space)
    else:
        raise f"Unknown type of agent {atype}.."

    env.inject(agent)

    draw_header()
    env.render()

    state = env.gen_obs()

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
                prev_state = agent.pos.copy()
                action = agent()

                actions += f" {env.actions.from_idx(action)}"
                if len(actions) > 80:
                    actions = actions[2 : len(actions) : 1]

                state, reward, done = env.step(action)

                if done == True:
                    stop = True
                    break

            draw_header()
            env.render()
            print(f"\033[1;34mIteration:\033[0m {env.n_steps:_d} /{env.step_max:_d}")
            print(f"\033[1;34mTotal    :\033[0m {agent.reward}", end=" ")
            print(
                f"[ from: {prev_state}, to: {env.world.agent.pos}, reward: {reward} ]"
            )
            print("\033[1;34mHistory  :\033[0m", end="")
            print(actions)

            n_steps = 1
