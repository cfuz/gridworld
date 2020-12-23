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
from coord import Coord
from action import Action


class QAgent(Agent):
    def __init__(
        self,
        action_space: numpy.array,
        n_states: int,
        x: int,
        y: int,
        discount: float,
        obey_factor: float,
        learning_rate: float,
    ):
        assert (
            discount >= 0.0 and discount <= 1.0
        ), f"Discount factor outside definition intervall ({gamma}).."
        assert (
            obey_factor >= 0.0 and obey_factor <= 1.0
        ), f"Best probability outside probability intervall ({p_best}).."
        assert (
            learning_rate >= 0.0
        ), f"Best probability outside probability intervall ({p_best}).."

        super().__init__(action_space, x, y, is_logic=True)

        # Defining the impact of future decisions in policy iteration
        self.discount = discount
        self.learning_rate = learning_rate

        self.q_values = numpy.zeros((n_states, self.n_actions), dtype=numpy.float32)
        self.state_policy = numpy.zeros(n_states, dtype=numpy.int32)

        # Defining randomness in decision
        self.p_obey = obey_factor
        self.p_disobey = (1.0 - self.p_obey) / (self.n_actions - 1)
        self.dist = (
            self.p_disobey * numpy.ones((self.n_actions, self.n_actions), dtype=float)
        ) + (self.p_obey - self.p_disobey) * numpy.eye(
            self.n_actions, self.n_actions, dtype=float
        )

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

        return (self.n_episodes, self.score, self.n_steps)

    def value(self, state: int or Coord = None) -> numpy.float32 or numpy.array:
        if state is not None:
            if isinstance(state, Coord):
                return self.q_values[state.to_state(), :].max()
            else:
                return self.q_values[state, :].max()
        else:
            return self.q_values.max(axis=1)

    def policy(self, state: int or Coord = None) -> numpy.int or numpy.array:
        if state is not None:
            if isinstance(state, Coord):
                return self.state_policy[state.to_state()]
            else:
                return self.state_policy[state]
        else:
            return self.state_policy

    def __call__(self, state: int or Coord) -> int:
        if isinstance(state, Coord):
            state = state.to_state()

        # Returns the best policy with probability self.p_obey
        return numpy.random.choice(
            self.action_space, p=self.dist[self.state_policy[state]]
        )
