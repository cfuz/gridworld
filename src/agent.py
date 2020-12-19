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

import coord

from cell import Cell


class Action(enum.Enum):
    North = coord.Coord(0, -1)
    East = coord.Coord(1, 0)
    South = coord.Coord(0, 1)
    West = coord.Coord(-1, 0)


class Agent(abc.ABC):
    """ Abstract agent """

    def __init__(self, grid: list, x_start: int = 0, y_start: int = 0):
        self.pos = coord.Coord(x_start, y_start)

        size = len(grid)
        self.rewards = {
            Cell.Empty: -1,
            Cell.Trap: -2 * (size - 1),
            Cell.Start: -1,
            Cell.Goal: 2 * (size - 1),
        }

        self.rgrid = numpy.array(
            [[self.rewards[col] for col in line] for line in grid], dtype=numpy.int32
        )

        # We pretend that the goal is represented as the max. value from the
        # reward grid.
        pos_rmax = numpy.where(self.rgrid == numpy.amax(self.rgrid))
        self.goal = coord.Coord(*pos_rmax)

        self.scan()

        self.reward = 0

    def scan(self):
        self.actions = []

        if self.pos != self.goal:
            if self.pos.x != 0:
                self.actions += [Action.West]
            if self.pos.x != len(self.rgrid) - 1:
                self.actions += [Action.East]
            if self.pos.y != 0:
                self.actions += [Action.North]
            if self.pos.y != len(self.rgrid) - 1:
                self.actions += [Action.South]

    def move(self, action: Action):
        self.pos += Agent._DIRECTION[action]

    @abc.abstractmethod
    def __call__(self):
        pass

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "🪖"


class RandomAgent(Agent):
    """ The world's simplest agent !"""

    def __init__(self, rgrid: list, x_start: int = 0, y_start: int = 0):
        super().__init__(rgrid, x_start=x_start, y_start=y_start)

    def __call__(self):
        return self.action_space.sample()

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()
