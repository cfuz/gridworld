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


# ref: https://www.youtube.com/watch?v=9g32v7bK3Co
class MdpAgent(Agent):
    def __init__(
        self,
        action_space: numpy.array,
        transitions: dict,
        x: int,
        y: int,
        discount: float,
        obey_factor: float,
    ):
        assert (
            discount >= 0.0 and discount <= 1.0
        ), f"Discount factor outside definition intervall ({gamma}).."
        assert (
            obey_factor >= 0.0 and obey_factor <= 1.0
        ), f"Best probability outside probability intervall ({p_best}).."

        super().__init__(action_space, x, y, is_logic=True)

        n_states = transitions["reward"].shape[0]

        # Defining the impact of future decisions in policy iteration
        self.discount = discount
        self.state_values = numpy.zeros(n_states, dtype=numpy.float32)
        self.state_policy = numpy.zeros(n_states, dtype=numpy.int32)

        # Epsilon greedy policy: defining randomness in decision
        self.p_obey = obey_factor
        self.p_disobey = (1.0 - self.p_obey) / (self.n_actions - 1)
        self.dist = (
            self.p_disobey * numpy.ones((self.n_actions, self.n_actions), dtype=float)
        ) + (self.p_obey - self.p_disobey) * numpy.eye(
            self.n_actions, self.n_actions, dtype=float
        )

        # Storing the 'rules'
        self.reward_transitions = numpy.copy(transitions["reward"])
        self.state_transitions = numpy.copy(transitions["next"])

        self.epsilon = 1.0e-5
        self.is_optimized = False

    def update(
        self,
        action: Action,
        reward: float,
        state: int or Coord,
        is_trap: bool,
        info: dict = None,
    ) -> (int, float, int, bool):
        def compute_action_value(
            state: int, action: int, next_states: numpy.array, state_value: numpy.array
        ) -> numpy.float32:
            # Computes the expected reward and value given the probability distribution
            expected_reward = (
                self.reward_transitions[state, :] * self.dist[action, :]
            ).sum()
            expected_value = (state_value[next_states] * self.dist[action, :]).sum()

            # Balancing the instantaneous expected reward with future value expectation
            return expected_reward + self.discount * expected_value

        def is_difference_substancial(
            old_state_value: numpy.array, new_state_value: numpy.array
        ) -> float:
            return numpy.linalg.norm(new_state_value - old_state_value) > self.epsilon

        if isinstance(state, Coord):
            state = state.to_state()

        if not self.is_optimized:
            old_state_values = numpy.copy(self.state_values)

            # Evaluating current policy
            for s, next_states in enumerate(self.state_transitions):
                self.state_values[s] = compute_action_value(
                    s, self.state_policy[s], next_states, old_state_values
                )

            # Improving policy
            for s, next_states in enumerate(self.state_transitions):
                for a in self.action_space:
                    action_value = compute_action_value(
                        s, a, next_states, old_state_values
                    )

                    if self.state_values[s] < action_value:
                        self.state_values[s] = action_value
                        self.state_policy[s] = a

            self.is_optimized = not is_difference_substancial(
                old_state_values, self.state_values
            )

        # Updating state
        super().update(action, reward, state, is_trap)

        return (self.n_episodes, self.score, self.n_steps)

    def value(self, state: int or Coord = None) -> numpy.float32 or numpy.array:
        if state is not None:
            if isinstance(state, Coord):
                return self.state_values[state.to_state()]
            else:
                return self.state_values[state]
        else:
            return self.state_values

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
