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


import abc
import enum
import numpy

from cell import Cell
from coord import Coord
from action import Action


class Agent(abc.ABC):
    def __init__(
        self, action_space: numpy.array, x: int, y: int, is_logic: bool = False
    ):
        self.is_logic = is_logic

        self.action_space = action_space
        self.n_actions = len(action_space)

        self.start_pos = Coord(x, y)
        self.pos = self.start_pos.copy()

        self.score = 0.0
        self.last_reward = 0.0
        self.last_action = None

        self.is_on_trap = False

        self.n_episodes = 0
        self.n_steps = 0
        self.steps = []
        self.scores = []

    def __repr__(self):
        return str(self)

    def __str__(self):
        elt = "🪖" if not self.is_on_trap else "💀"
        if self.last_action:
            if self.last_action == Action.West:
                elt = f"{self.last_action}" + elt
            else:
                elt += f"{self.last_action}"
        return elt

    def reset(self):
        self.scores += [self.score]
        self.steps += [self.n_steps]
        self.n_episodes += 1

        self.score = 0.0
        self.n_steps = 0

        self.pos = self.start_pos.copy()

        self.last_reward = 0.0
        self.last_action = None

        self.is_on_trap = False

    def history(self) -> list:
        return list(zip(self.scores, self.steps))

    @abc.abstractmethod
    def update(self, action: Action, reward: float, state: int or Coord, is_trap: bool):
        self.pos = state if isinstance(state, Coord) else Coord.from_state(state)

        self.score += reward
        self.n_steps += 1

        self.last_reward = reward
        self.last_action = action

        if is_trap:
            self.is_on_trap = True
        else:
            self.is_on_trap = False

    @abc.abstractmethod
    def __call__(self, state: int or Coord) -> int:
        pass

    @abc.abstractmethod
    def value(self, state: int or Coord = None):
        pass

    @abc.abstractmethod
    def policy(self, state: int or Coord = None):
        pass
