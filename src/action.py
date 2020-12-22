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


import enum
import numpy

from coord import Coord


class Action(enum.Enum):
    North = Coord(0, -1)
    East = Coord(1, 0)
    South = Coord(0, 1)
    West = Coord(-1, 0)

    @staticmethod
    def to_indices() -> numpy.array:
        return numpy.arange(len(Action), dtype=numpy.int32)

    @staticmethod
    def from_idx(action_idx: int):
        assert action_idx >= 0 and action_idx <= 3, "Action index out of bound.."

        if action_idx == 0:
            return Action.North
        elif action_idx == 1:
            return Action.East
        elif action_idx == 2:
            return Action.South
        else:
            return Action.West

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self == Action.North:
            return ""
        elif self == Action.East:
            return ""
        elif self == Action.South:
            return ""
        else:
            return ""
