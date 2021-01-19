#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "2.0.0"
__maintainer__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]


import enum
import numpy

from coord import Coord


class Action(enum.Enum):
    """
    `Action` defines the set of actions available in the gridworld problem.
    """

    North = Coord(0, -1)
    East = Coord(1, 0)
    South = Coord(0, 1)
    West = Coord(-1, 0)

    @staticmethod
    def to_indices() -> numpy.array:
        """
        Convert the set of actions to an array of integers

        Returns
        -------
        (numpy.array):
            The list of action indices.
        """
        return numpy.arange(len(Action), dtype=numpy.int32)

    @staticmethod
    def from_idx(action_idx: int):
        """
        Get the appropriate Action variant from its index

        Parameter
        ---------
        action_idx (int):
            Index of the action to retrieve

        Returns
        -------
        (Action):
            The corresponding action
        """
        assert action_idx >= 0 and action_idx <= 3, "Action index out of bound.."

        if action_idx == 0:
            return Action.North
        elif action_idx == 1:
            return Action.East
        elif action_idx == 2:
            return Action.South
        else:
            return Action.West

    def to_idx(self) -> int:
        """
        Converts the current `Action`'s instance to its index

        Returns
        -------
        (int):
            The corresponding Action's index
        """
        if self == Action.North:
            return 0
        elif self == Action.East:
            return 1
        elif self == Action.South:
            return 2
        else:
            return 3

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self == Action.North:
            return ""
        elif self == Action.East:
            return ""
        elif self == Action.South:
            return ""
        else:
            return ""
