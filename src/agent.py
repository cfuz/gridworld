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


class Agent(abc.ABC):
    """ Abstract agent """

    def __init__(self, action_space: numpy.array):
        self.action_space = action_space

    @abc.abstractmethod
    def __call__(self, state: numpy.array):
        pass
