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


import enum


class Cell(enum.Enum):
    Empty = 0
    Start = 1
    Trap = 2
    Goal = 3

    def __str__(self):
        if self == Cell.Empty:
            return ""
        elif self == Cell.Start:
            return "🪧"
        elif self == Cell.Trap:
            return "🔥"
        else:
            return "📦"

    def __repr__(self):
        return str(self)
