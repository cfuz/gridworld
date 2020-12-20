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
import numpy as np

import coord

from cell import Cell


class Agent(abc.ABC):
    """ Abstract agent """

    def __init__(self, action_space: np.array):
        self.action_space = action_space

    @abc.abstractmethod
    def __call__(self, state: np.array):
        pass


class RandomAgent(Agent):
    """ The world's simplest agent !"""
    def __init__(self, action_space: np.array, cfg: dict = None):
        super().__init__(action_space)
        self.cfg = cfg

    def __call__(self, state: np.array):
        return self.action_space.sample()


class StairsAgent(Agent):
    """ The world's simplest agent !"""
    def __init__(self, action_space: np.array, cfg: dict = None):
        super().__init__(action_space)
        self.cfg = cfg
        self.down = False

    def __call__(self, state: np.array):
        self.down = not self.down
        return 1 if self.down else 2
