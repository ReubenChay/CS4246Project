from typing import Callable
from utils.Agent import MCAgent
from utils.MazeEnv import MazeEnvProblem, Maze
from utils.configs import walls_maze1
import random
import numpy as np



class MCT_Node:
    """Node in the Monte Carlo search tree, keeps track of the children states."""

    def __init__(self, parent=None, act=None, state=None, record=None, U=0, N=0):
        self.__dict__.update(parent=parent, state=state, U=U, N=N)
        self.children = {}
        self.record = record
        self.act = act


class MCTSAgent(MCAgent):
    def __init__(self, problem):
        super().__init__(problem)
        self.N = 20
        self.maze = problem.maze
        self.record = np.zeros(self.maze.state_shape)
        self.result = {}

    def ucb(self, n, C=1.4):
        ucb = np.inf if n.N == 0 else n.U / n.N + C * np.sqrt(np.log(n.parent.N) / n.N)
        return ucb

    def takeaction(self, act, record):
        new = record.copy()
        new[act] += 1
        return new

    def search(self, s, goal_test: Callable, actions: Callable, is_terminal: Callable,
               result: Callable, utility: Callable):
        if goal_test(s):
            a = None
            return a

        record = self.record.copy()
        self.record[s] += 1
        root = MCT_Node(state=s, record=self.takeaction(s, record))

        for _ in range(self.N):

            child = self.expand(self.select(root, actions), result, actions)
            self.backprop(child, self.simulate(child, actions, is_terminal, result, utility))


        max_state = max(root.children, key=lambda p: p.N)

        return max_state.act

    def select(self, n, actions: Callable):
        """select a leaf node in the tree"""
        if n.children:

            return self.select(max(n.children, key = self.ucb), actions)
        
        else:
            return n

    def expand(self, n, result: Callable, actions: Callable):
        """expand the leaf node by adding all its children states"""

        for action in actions(n.state, n.record):
            n.children[MCT_Node(parent=n, act=action, state=result(n.state, action), record = self.takeaction(n.state, n.record))] = None

        return self.select(n, actions)

    def simulate(self, child, actions: Callable, is_terminal: Callable, result: Callable, utility: Callable):
        """simulate the utility of current state by random picking a step"""
        cost = 1
        state = child.state
        record = child.record
        while not is_terminal(state, record):
            action = random.choice(list(actions(state, record)))
            state = result(state, action)
            record = self.takeaction(state, record)
            cost += 1
        v = utility(state, cost)
        return v

    def backprop(self, n, utility):
        """passing the utility back to all parent nodes"""
        if utility > 0:
            n.U += utility
        n.N += 1
        if n.parent:
            self.backprop(n.parent, utility)



def print_path(path):
    for p in path:
        action = p[-1]
        if action is not None:
            print("state: {}, action: {}".format(p[0], action))
        else:
            print("state: {}".format(p[0]))


def run(problem, num_steps: int):
    path = []
    # Initialize mazes and problems objects

    agent = MCTSAgent(problem)
    state = problem.initial
    for _ in range(num_steps):
        a = agent(state)
        path.append((state, a))
        if a is None:
            break
        state_next = problem.result(state, a)
        state = state_next
    return path

wall_setup = walls_maze1
maze1 = Maze(width = len(wall_setup[0]), height = len(wall_setup), walls = wall_setup)
problem1 = MazeEnvProblem(initial=(0, ) * (maze1.width + 2), goal=(maze1.width -1, maze1.width - 1), maze=maze1)

path = run(problem1, 100)
print_path(path)
