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


import abc
import enum
import numpy

from cell import Cell
from coord import Coord
from action import Action


class Agent(abc.ABC):
    """
    `Agent` is an abstract class that fixes the global parameters and logic 
    for our decision making process.

    Attributes
    ----------
    is_logic (bool):
        Defines if the concrete agent implementation contains logic methods to
        update its weights
    n_actions (int):
        Number of available actions
    start_pos (Coord):
        Starting position
    pos (Coord):
        Current agent's position
    score (float):
        Agent's current session score
    last_reward (float):
        Agent's last reward
    last_action (Action):
        Agent's last action
    is_on_trap (bool):
        Set to True if the Agent is currently on a trap (for display purposes)
    n_episodes (int):
        Number of episodes run by the concrete agent's instance
    n_steps (int):
        Number of steps done for current epoch
    steps (list):
        History of the number of steps made for each epoch
    scores (list):
        History of the scores from previous epochs
    """

    def __init__(
        self, action_space: numpy.array, x: int, y: int, is_logic: bool = False
    ):
        self.is_logic = is_logic

        self.action_space = action_space
        self.n_actions = len(action_space)

        self.start_pos = Coord(x, y)
        self.pos = self.start_pos.copy()

        self.score = 0.0
        self.last_reward = 0.0
        self.last_action = None

        self.is_on_trap = False

        self.n_episodes = 0
        self.n_steps = 0
        self.steps = []
        self.scores = []

    def reset(self):
        """
        Reset `Agent` to its initial state.
        """
        self.scores += [self.score]
        self.steps += [self.n_steps]
        self.n_episodes += 1

        self.score = 0.0
        self.n_steps = 0

        self.pos = self.start_pos.copy()

        self.last_reward = 0.0
        self.last_action = None

        self.is_on_trap = False

    def history(self) -> list:
        """
        Retrieves the history of the `Agent`.

        Returns
        -------
        (list[(float, int)]):
            A list of couples, giving for each epoch, the score and the number 
            of steps made by the agent.
        """
        return list(zip(self.scores, self.steps))

    @abc.abstractmethod
    def update(self, action: Action, reward: float, state: int or Coord, is_trap: bool):
        """
        Update generic `Agent`'s attribute given its last past actions.

        Parameters
        ----------
        action (Action):
            Action taken
        reward (float):
            Reward obtained from action taken
        state (int or Coord):
            Landing state
        is_trap (bool):
            If the landing state is a trap (True) or not (False)
        """
        self.pos = state if isinstance(state, Coord) else Coord.from_state(state)

        self.score += reward
        self.n_steps += 1

        self.last_reward = reward
        self.last_action = action

        if is_trap:
            self.is_on_trap = True
        else:
            self.is_on_trap = False

    @abc.abstractmethod
    def __call__(self, state: int or Coord) -> int:
        """
        Computes the next action taken by the `Agent`'s concrete instance given 
        a `state`.

        Parameter
        ---------
        state (int or Coord):
            The state at which the agent should compute the next decision
        
        Returns
        -------
        (int):
            The index of the action taken
        """
        pass

    @abc.abstractmethod
    def value(self, state: int or Coord = None):
        pass

    @abc.abstractmethod
    def policy(self, state: int or Coord = None):
        pass

    def __repr__(self):
        return str(self)

    def __str__(self):
        elt = "🪖" if not self.is_on_trap else "💀"
        if self.last_action:
            if self.last_action == Action.West:
                elt = f"{self.last_action}" + elt
            else:
                elt += f"{self.last_action}"
        return elt
