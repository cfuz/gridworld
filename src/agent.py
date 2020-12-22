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


class Agent(abc.ABC):
    def __init__(
        self, action_space: numpy.array, x: int, y: int, is_logic: bool = False
    ):
        self.action_space = action_space
        self.n_actions = len(action_space)

        self.pos = Coord(x, y)

        self.is_logic = is_logic

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "🪖"

    @abc.abstractmethod
    def __call__(self, state: int):
        pass

    @abc.abstractmethod
    def value(self, state: int or Coord = None):
        pass

    @abc.abstractmethod
    def policy(self, state: int or Coord = None):
        pass
