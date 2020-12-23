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


import numpy

from agent import Agent


class SarsaAgent(Agent):
    def __init__(self, action_space: numpy.array, cfg: dict = None):
        super().__init__(action_space)
        self.cfg = cfg

    def __call__(self, state: numpy.array):
        return self.action_space.sample()
