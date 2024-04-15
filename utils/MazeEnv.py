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
            return state[:index + 2] + (state[index + 2] - 1,) + state[index+3:]
        else:
            return state[:index + 2] + (state[index + 2] + 1,) + state[index+3:]



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
        self.action_matrix = np.ones(action_shape)
        self.walls = np.zeros((width, height, 4))
        for i in width:
            for j in height:
                for idx in range(4):
                    if (walls[i, j]&(1<<idx)) != 0:
                        self.walls[i, j, idx] = True


        self.generate_action_matrix()

    def generate_action_matrix(self):
        """
        The function to generate a matrix in size of (X,Y,A) to show available actions.

        0 - moving upwards
        1 - moving downwards
        2 - moving leftwards
        3 - moving rightwards
        """


        # + delete all actions that make the agent go out of the maze.
        self.action_matrix[:, self.height - 1, 0] = 0
        self.action_matrix[:, 0, 1] = 0
        self.action_matrix[0, :, 2] = 0
        self.action_matrix[self.width - 1, :, 3] = 0

        # + delete all actions that make the agent move against the walls
        for wall in self.walls:
            if wall[0][0] == wall[1][0]:
                # + for horizon walls, delete the 0 action lower than it and 1 action upper than it.
                self.action_matrix[wall[0][0], wall[0][1], 0] = 0
                self.action_matrix[wall[1][0], wall[1][1], 1] = 0
            elif wall[0][1] == wall[1][1]:
                # + for vertical walls, delete the 2 action right to it and 3 action left to it.
                self.action_matrix[wall[0][0], wall[0][1], 3] = 0
                self.action_matrix[wall[1][0], wall[1][1], 2] = 0

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
