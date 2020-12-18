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


class Action(enum.Enum):
    North = 0
    East = 1
    South = 2
    West = 3


class Agent(abc.ABC):
    """ Abstract agent """

    def __init__(self, x_start: int = 0, y_start: int = 0, cfg: dict = None):
        self.x = x_start
        self.y = y_start
        self.reward = 0

        if cfg is not None:
            self.cfg = cfg

    def scan(self, grid: list):
        self.limit = [0, len(grid)]

    @abc.abstractmethod
    def __call__(self, state: numpy.array):
        pass


class RandomAgent(Agent):
    """ The world's simplest agent !"""

    def __init__(self, x_start: int = 0, y_start: int = 0, cfg: dict = None):
        super().__init__(x_start=x_start, y_start=y_start, cfg=cfg)

    def __call__(self, state: numpy.array):
        return self.action_space.sample()
