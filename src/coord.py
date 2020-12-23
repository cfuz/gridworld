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


class Coord:
    _SIZE = None

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def copy(self):
        return type(self)(self.x, self.y)

    @staticmethod
    def from_state(state: int):
        assert (
            Coord._SIZE is not None
        ), "Coord struct needs to have its static attribute _WIDTH set.."
        return Coord(state % Coord._SIZE, state // Coord._SIZE)

    def to_state(self) -> int:
        assert (
            Coord._SIZE is not None
        ), "Coord struct needs to have its static attribute _WIDTH set.."
        return self.y * Coord._SIZE + self.x

    def __add__(self, other):
        x = min(max(self.x + other.x, 0), Coord._SIZE - 1)
        y = min(max(self.y + other.y, 0), Coord._SIZE - 1)
        return Coord(x, y)

    def __eq__(self, other):
        if isinstance(other, Coord):
            return self.x == other.x and self.y == other.y
        return False

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"({self.x},{self.y})"
