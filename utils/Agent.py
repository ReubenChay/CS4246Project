"""
The script to define agents.
"""

from typing import Callable


class Agent:
    def __init__(self, problem):
        self.problem = problem

    def __call__(self, s):
        actions_available = self.problem.actions(s)
        a = self.searching(s, actions_available, self.problem.result, self.problem.value)
        return a

    def searching(self, s, actions_available, result: Callable, v: Callable):
        raise NotImplementedError



class MCAgent(Agent):
    """
    The online-search algorithm agent.
    """

    def __init__(self, problem):
        super(MCAgent, self).__init__(problem)
        self.s = None
        self.a = None

    def __call__(self, s):
        action = self.search(s, self.problem.goal_test, self.problem.actions, 
                             self.problem.is_terminal, self.problem.result, self.problem.utility)
        self.s = s
        self.a = action
        return action

    def ucb(self, n):
        raise NotImplementedError

    def select(self, n):
        raise NotImplementedError

    def expand(self, n):
        raise NotImplementedError

    def simulate(self, game, state):
        raise NotImplementedError

    def backprop(self, n, utility):
        raise NotImplementedError

