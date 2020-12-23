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
from agent import Agent
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
\033[1;33mGRIDWORLD {__version__}
\033[0;33m{GRID_CFG}{AGENT_CFG}\033[0m
    """
    )


def draw_state(env: env.GridWorld, agent: Agent, prev_state: Coord, actions: str):
    draw_header()
    env.render()
    print(f"\033[1;34mIteration:\033[0m {agent.n_steps:_d} /{env.step_max:_d}")
    print(
        f"""\
\033[1;34mTotal    :\033[0m {agent.score} \
[ from: {prev_state}, to: {env.world.agent.pos}, reward: {agent.last_reward} ]\
    """
    )
    print(f"\033[1;34mHistory\033[0m")
    print(f"  \033[0;34mLast actions:\033[0m {actions}")
    print("  \033[0;34mBenchmark\033[0m")
    for episode_idx, (score, steps) in enumerate(agent.history()):
        print(
            f"    \033[1m#{episode_idx:<3d}\033[0m score: {score:>+8.1f}, steps: {steps}"
        )


def run_steps(
    n_steps: int, env: env.GridWorld, agent: Agent, actions: str, reset: bool
) -> (str, Coord):
    if reset:
        agent.reset()

    state = agent.pos.copy()
    for _ in range(n_steps):
        prev_state = agent.pos.copy()

        action = agent(env.gen_obs())
        action = env.actions.from_idx(action)
        actions += f" {action}"
        if len(actions) > 80:
            actions = actions[2 : len(actions) : 1]

        state, reward, done, is_trap = env.step(prev_state, action)

        n_episode, score, n_steps, _optimized = agent.update(
            action, reward, state, is_trap
        )

        if done:
            break

    return (actions, prev_state, done)


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
    if "traps" in cfg["world"]:
        if cfg["world"]["traps"]["type"] == "random":
            GRID_CFG = f'RANDOM  DENSITY: {cfg["world"]["traps"]["dist"]["trap"]}   '
        elif cfg["world"]["traps"]["type"] == "fixed":
            GRID_CFG = f"FIXED   "
    else:
        GRID_CFG = f"NOTRAP   "

    atype = cfg["agent"]["type"].lower()
    if atype in ["random", "rand"]:
        AGENT_CFG = "RANDAGT"
        agent = RandomAgent(env.actions.to_indices(), **cfg["world"]["start"])
    elif atype == "mdp":
        AGENT_CFG = f'MDPAGT  DISC: {cfg["agent"]["discount"]}  OBEY: {cfg["agent"]["obey_factor"]}'
        del cfg["agent"]["type"]
        agent = MdpAgent(
            env.actions.to_indices(),
            env.transitions,
            **cfg["world"]["start"],
            **cfg["agent"],
        )
    elif atype == "sarsa":
        AGENT_CFG = f'SARSAAGT  DISC: {cfg["agent"]["discount"]}  OBEY: {cfg["agent"]["obey_factor"]}  LR: {cfg["agent"]["learning_rate"]}'
        del cfg["agent"]["type"]
        agent = SarsaAgent(
            env.actions.to_indices(),
            env.transitions,
            **cfg["world"]["start"],
            **cfg["agent"],
        )
    # elif atype == "q":
    #     agent = QAgent(env.action_space)
    else:
        raise f"Unknown type of agent {atype}.."

    env.inject(agent)

    draw_header()
    env.render()

    state = env.gen_obs()

    stop = False
    done = False
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
            actions, prev_state, done = run_steps(n_steps, env, agent, actions, done)
            # if agent.optimized:
            #     agent.reset()

            # for _ in range(n_steps):
            #     prev_state = agent.pos.copy()

            #     action = agent(env.gen_obs())
            #     action = env.actions.from_idx(action)
            #     actions += f" {action}"
            #     if len(actions) > 80:
            #         actions = actions[2 : len(actions) : 1]

            #     state, reward, done, is_trap = env.step(action)

            #     n_episode, score, n_steps, optimized = n_agent.update(state, action, reward, is_trap)

            #     if optimized or done:
            #         break

            draw_state(env, agent, prev_state, actions)

            n_steps = 1
