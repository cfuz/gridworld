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


# class Agent(abc.ABC):
#     """ Abstract agent """

#     _DIRECTION = {
#         Action.North: coord.Coord(0, -1),
#         Action.East: coord.Coord(1, 0),
#         Action.South: coord.Coord(0, 1),
#         Action.West: coord.Coord(-1, 0),
#     }

#     def __init__(self, rgrid: list, x_start: int = 0, y_start: int = 0):
#         self.pos = coord.Coord(x_start, y_start)

#         self.rgrid = numpy.array(rgrid, dtype=numpy.int32)

#         # We pretend that the goal is represented as the max. value from the
#         # reward grid.
#         pos_rmax = numpy.where(self.rgrid == numpy.amax(self.rgrid))
#         self.goal = coord.Coord(*pos_rmax)

#         self.actions = []

#         self.reward = 0

#     def scan(self, grid: list):
#         self.actions = []

#         if self.pos != self.goal:
#             if self.pos.x != 0:
#                 self.actions += [Action.West]
#             if self.pos.x != len(self.rgrid) - 1:
#                 self.actions += [Action.East]
#             if self.pos.y != 0:
#                 self.actions += [Action.North]
#             if self.pos.y != len(self.rgrid) - 1:
#                 self.actions += [Action.South]

#     def move(self, action: Action):
#         self.pos += Agent._DIRECTION[action]

#     @abc.abstractmethod
#     def __call__(self):
#         pass