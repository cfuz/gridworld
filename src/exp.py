#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.0.0"
__maintainer__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]

import argparse
import numpy
import yaml

import env
from agent import Agent
from coord import Coord
from action import Action

class ValueIteration(Agent):
    def __init__(
        self,
        action_space: numpy.array,
        transitions: dict,
        x: int,
        y: int
    ):
        super().__init__(action_space, x, y, is_logic=True)
        n_states = transitions["reward"].shape[0]
        self.state_values = numpy.zeros(n_states, dtype=numpy.float32)

    def update(
        self, action: Action, reward: float, state: int or Coord, is_trap: bool,
    ) -> (int, float, int, bool):
        pass

    def value(self, state: int or Coord = None) -> numpy.float32 or numpy.array:
        if state is not None:
            if isinstance(state, Coord):
                return self.state_values[state.to_state()]
            else:
                return self.state_values[state]
        else:
            return self.state_values

    def policy(self, state: int or Coord = None) -> numpy.int or numpy.array:
        pass

    def __call__(self, state: int or Coord) -> int:
        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Gridworld with different RL algorithms"
    )
    parser.add_argument("cfg", type=str, help="Path to .yaml configuration file")
    args = parser.parse_args()

    with open(args.cfg, "r") as file:
        cfg = yaml.safe_load(file)

    env = env.GridWorld(cfg)

    agent = ValueIteration(            
        env.actions.to_indices(),
        env.transitions,
        **cfg["world"]["start"],
    )

    env.inject(agent)
    env.render()