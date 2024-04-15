"""
Script for searching problems.
"""


class Problem:
    """The abstract class for a formal problem."""

    def __init__(self, initial, goal):
        """The constructor specifies the initial state, and a goal state."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given state."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given action in the given state."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal."""
        return state[0] == self.goal[0] and state[1] == self.goal[1]


    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError
