"""
The script to define the maze environments.
"""

import numpy as np

from utils.SearchBase import Problem


class MazeEnvProblem(Problem):
    """
    The class to define a maze environment.
    """

    def __init__(self, initial, goal, maze):
        """
        init function.
        Each state is a location (x,y).
        :param initial: initial state.
        :param goal: goal state.
        :
        """
        super(MazeEnvProblem, self).__init__(initial, goal)
        self.maze = maze

    def actions(self, state, record=None):
        """
        :param state: the current state.
        :return: available actions at state.
        """
        actions = self.maze.available_actions(state)
        if record is None:
            return actions
        else:
            acts = []
            for action in actions:
                if record[self.result(state, action)] == 0:
                    acts.append(action)
            return acts

    def result(self, state, action):
        """
        Return the state that results from executing the given action in the given state.
        :param state: the given state.
        :param action: the given action.
        """
        # check availability of the action.
        assert self.maze.check_available_action(state, action), \
            "The action {} is NOT available at state {}".format(action, state)

        if action == 0:
            # 0 - moving upwards
            return (state[0], state[1] + 1) + state[2:]
        if action == 1:
            # 1 - moving downwards
            return (state[0], state[1] - 1) + state[2:]
        if action == 2:
            # 2 - moving leftwards
            return (state[0] - 1,) + state[1:]
        if action == 3:
            # 3 - moving rightwards
            return (state[0] + 1,) + state[1:]
        index = (action - 4)//2
        if (action - 4)%2:
            return state[:index + 2] + ((state[index + 2] - 1) % self.maze.height,) + state[index+3:]
        else:
            return state[:index + 2] + ((state[index + 2] + 1) % self.maze.height,) + state[index+3:]



    def is_terminal(self, state, record):
        """Return True if this is a final state for the game."""
        return not self.actions(state, record) or (state[0] == self.goal[0] and state[1] == self.goal[1])

    def utility(self, state, cost):
        """Return the value of this final state to player."""
        if state[0] == self.goal[0] and state[1] == self.goal[1]:
            return 1 / cost
        else:
            return 0


class Maze:
    """
    The class to define a maze.
    """

    def __init__(self, width, height, walls):
        """
        init function.
        """

        self.width = width
        self.height = height
        self.state_shape = tuple([width] + [height] * (width + 1))
        action_shape = tuple([width] + [height] * (width + 1) + [4 + 2 * width])
        self.action_matrix = np.zeros(action_shape)
        self.walls = np.zeros((width, height, 4))
        for i in range(width):
            for j in range(height):
                for idx in range(4):
                    if (walls[i][j]&(1<<idx)) != 0:
                        self.walls[(i, j, idx)] = True


        self.generate_action_matrix()

    def generate_action_matrix(self):
        """
        The function to generate a matrix in size of (X,Y,A) to show available actions.

        0 - moving upwards
        1 - moving downwards
        2 - moving leftwards
        3 - moving rightwards
        """
        
        
        self.action_matrix[..., :4] = 1

        # + delete all actions that make the agent go out of the maze.
        self.action_matrix[(slice(None),) + (self.height - 1,) + (slice(None),) * self.width + (0,)] = 0
        self.action_matrix[(slice(None),) + (0,) + (slice(None),) * self.width + (1,)] = 0
        self.action_matrix[(0,) + (slice(None),) * (self.width + 1) + (2,)] = 0
        self.action_matrix[(self.width - 1,) + (slice(None),) * (self.width + 1) + (3,)] = 0


        for i in range(self.width): #bot x
            for j in range(self.height): #bot y
                for col in range(self.height): #col shift at col x
                    for dir in range(4):
                        if self.walls[i][col][dir]:
                            self.action_matrix[(i, j) + (slice(None),) * i + ((j - col) % self.height,) + (slice(None),) * (self.width - i -1) + (dir,)] = 0

        for i in range(self.width - 1): #bot x
            for j in range(self.height): #bot y
                for col in range(self.height): #col shift at col x + 1
                    if self.walls[i + 1][col][2]:
                        self.action_matrix[(i, j) + (slice(None),) * i + ((j - col) % self.height,) + (slice(None),) * (self.width - i -1) + (3,)] = 0

        for i in range(1, self.width): #bot x
            for j in range(self.height): #bot y
                for col in range(self.height): #col shift at col x - 1
                    if self.walls[i - 1][col][3]:
                        self.action_matrix[(i, j) + (slice(None),) * i + ((j - col) % self.height,) + (slice(None),) * (self.width - i -1) + (2,)] = 0


        for i in range(self.width - 1): #bot x
            self.action_matrix[i, ..., 2 * i + 6] = 1
            self.action_matrix[i, ..., 2 * i + 7] = 1
        for i in range(1, self.width): #bot x
            self.action_matrix[i, ..., 2 * i + 2] = 1
            self.action_matrix[i, ..., 2 * i + 3] = 1

    def check_available_action(self, state, action):
        """
        The function to check whether the action is available in state.
        :param state: the current state.
        :param action: the possible action.
        :return: True for available, False for unavailable.
        """
        return True if self.action_matrix[state + (action,)] else False

    def available_actions(self, state):
        """
        The function to return available actions for the given state.
        :param state: the given state.
        :return: the list of available actions.
        """
        return np.argwhere(self.action_matrix[state] == 1).flatten().tolist()
