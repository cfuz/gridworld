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


class RandomAgent(Agent):
    def __init__(
        self, action_space: numpy.array, x: int, y: int,
    ):
        super().__init__(action_space, x, y, is_logic=False)

    def update(
        self, action: Action, reward: float, next_state: int or Coord, is_trap: bool,
    ) -> (int, float, int):
        if isinstance(next_state, Coord):
            next_state = next_state.to_state()

        state = self.pos.to_state()
        aidx = action.to_idx()

        best_action = self.q_values[next_state, :].argmax()
        target_value = reward + self.discount * self.q_values[next_state, best_action]
        delta = target_value - self.q_values[state, aidx]
        self.q_values[state, aidx] += self.learning_rate * delta
        self.state_policy[state] = self.q_values[state, :].argmax()

        # Updating agent's state
        super().update(action, reward, next_state, is_trap)

    def __call__(self, state: numpy.array):
        return numpy.random.choice(self.action_space)
