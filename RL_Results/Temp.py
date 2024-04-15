import sys

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
opp_dir = [DOWN, UP, LEFT, RIGHT]
delta = [(-1,0), (1,0), (0,1), (0,-1)]
#card type is denoted by a 4 bit number.
#For example 0101 means a wall in down and a wall in right
class Card:
    def __init__(self, no):
        self.walls = [False, False, False, False]
        self.number = no
        for idx in range(4):
            if (no&(1<<idx)) != 0:
                self.walls[idx] = True



board1 = [[1,0,4], [0,15,0], [0,8,8]]

import random
def generate_random_board(n, m):
    return [[random.randint(0, 15) for _ in range(m)] for _ in range(n)]


def read_from_file(filename):
    f = open(filename, 'r')
    arr = []
    for line in f:
        arr.append(list(map(int, line.split())))
    f.close()
    ## check if its valid
    for i in range(len(arr)):
        assert len(arr[0]) == len(arr[i])
        for j in range(len(arr[i])):
            assert(arr[i][j] in range(0, 16))
    ##
    return arr


class Board_CYCLEROWONLY:
    def __init__(self, card_no, max_moves = 100):
        self.n = len(card_no)
        self.m = len(card_no[0])
        self.board = [[Card(card_no[i][j]) for j in range(self.m)] for i in range(self.n)]
        self.robot_r = 0
        self.robot_c = 0
        self.original_state = card_no
        self.max_moves = max_moves

        self.moves_used = 0


    def in_bound(self, r, c):
        return r in range(0, self.n) and c in range(0, self.m)

    def cycle_col(self, c, offset):
        if self.robot_c == c:
            self.robot_r = (self.robot_r + self.n + offset)%self.n

            # return False
        nw_cards = []
        for i in range(0, self.n):
            nw_cards.append(self.board[(i - offset + self.n)%self.n][c])
        for i in range(0, self.n):
            self.board[i][c] = nw_cards[i]
        return True

    # def cycle_row(self, r, offset):
    #     if self.robot_r == r:
    #         return False
    #     nw_cards = []
    #     for j in range(0, self.m):
    #         nw_cards.append(self.board[r][(j - offset + self.m)%self.m])
    #     for j in range(0, self.m):
    #         self.board[r][j] = nw_cards[j]
    #     return True

    def move(self, dir):
        (dr, dc) = delta[dir]
        nr = self.robot_r + dr
        nc = self.robot_c + dc
        # check if the next cell is in bound and there is no walls linking you and no walls from the new cells
        if (not self.in_bound(nr, nc)
            or self.board[self.robot_r][self.robot_c].walls[dir]
            or self.board[nr][nc].walls[opp_dir[dir]]):
            return False
        self.robot_r = nr
        self.robot_c = nc
        return True

    def number_of_actions(self):
      return self.m*2 + 4
    def number_of_input_dims(self):
      return self.n*self.m + 2
    ## A card will look like this
    ## #?#
    ## ? ?
    ## #?#
    ## ? will be X if there is a wall, else it will be empty space
    def __repr__(self):
        str_arr = [[' ' for _ in range(self.m*3)] for _ in range(self.n * 3)]
        for i in range(self.n):
            for j in range(self.m):
                for (di, dj) in [(0,0), (2,0), (0,2), (2,2)]:
                    str_arr[i*3 + di][j*3 + dj] = '#'
                for dir in range(4):
                    if self.board[i][j].walls[dir]:
                        (di, dj) = delta[dir]
                        str_arr[i*3 + 1 + di][j*3 + 1 + dj] = 'X'

        str_arr[self.robot_r * 3 + 1][self.robot_c * 3 + 1] = 'R'
        ss = ""
        for i in range(len(str_arr)):
            for j in range(len(str_arr[0])):
                ss += str_arr[i][j]
            ss += "\n"
        return ss
    def print_state(self):
        print(self)

    def get_state_repr(self):
        arr = [self.robot_r, self.robot_c]
        for i in range(self.n):
          for j in range(self.m):
            arr.append(self.board[i][j].number)
        return arr



  ## ACTIONS THAT ROBOT CAN DO (true means succeed, false means not succeed)
      # def cycle_left(self, r):
      #     return self.cycle_row(r, 1)
      # def cycle_right(self, r):
      #     return self.cycle_row(r, -1)
    def cycle_down(self, c):
        return (self.cycle_col(c, 1), "CYCLE DOWN: " + str(c))
    def cycle_up(self, c):
        return (self.cycle_col(c, -1), "CYCLE UP: " + str(c))
    def move_down(self):
        return (self.move(DOWN), "MOVE DOWN")
    def move_up(self):
        return (self.move(UP), "MOVE UP")
    def move_left(self):
        return (self.move(LEFT), "MOVE LEFT")
    def move_right(self):
        return (self.move(RIGHT), "MOVE RIGHT")

      ## 0->3 are movements, 4->4 + (m-1) are cycle down, 4+m->4 + (2m-1) are cycle ups
    def do_action(self, action_no):
      if action_no == 0:
        return self.move_down()
      if action_no == 1:
        return self.move_up()
      if action_no == 2:
        return self.move_left()
      if action_no == 3:
        return self.move_right()
      if action_no < 4 + self.m:
        return self.cycle_down(action_no - 4)
      else:
        return self.cycle_up(action_no - 4 - self.m)

    def claim_victory(self):
        if self.robot_c == self.m - 1 and self.robot_r == self.n - 1 and not self.board[-1][-1].walls[DOWN]:
            return True
        return False

    # may need to reset board to start state
    def reset(self):
        self.board = [[Card(self.original_state[i][j]) for j in range(self.m)] for i in range(self.n)]
        self.robot_c = 0
        self.robot_r = 0
        self.moves_used = 0

    # update board after action taken
    def step(self, action):
        (able, info) = self.do_action(action)
        reward = -1.0
        is_done = False
        self.moves_used += 1
        if self.claim_victory():
          reward = 1000.0
          is_done = True
        else:
          #taking invalid moves waste moves...
          if not able:
            reward = -5.0
          if self.moves_used == self.max_moves:
            is_done = True
        return self.get_state_repr(), reward, is_done, info


class Board:
    def __init__(self, card_no, max_moves = 100):
        self.n = len(card_no)
        self.m = len(card_no[0])
        self.board = [[Card(card_no[i][j]) for j in range(self.m)] for i in range(self.n)]
        self.robot_r = 0
        self.robot_c = 0
        self.original_state = card_no
        self.max_moves = max_moves

        self.moves_used = 0


    def in_bound(self, r, c):
        return r in range(0, self.n) and c in range(0, self.m)

    def cycle_col(self, c, offset):
        if self.robot_c == c:
            #self.robot_r = (self.robot_r + self.n + offset)%self.n
             return False
        nw_cards = []
        for i in range(0, self.n):
            nw_cards.append(self.board[(i - offset + self.n)%self.n][c])
        for i in range(0, self.n):
            self.board[i][c] = nw_cards[i]
        return True

    def cycle_row(self, r, offset):
        if self.robot_r == r:
            return False
        nw_cards = []
        for j in range(0, self.m):
            nw_cards.append(self.board[r][(j - offset + self.m)%self.m])
        for j in range(0, self.m):
            self.board[r][j] = nw_cards[j]
        return True

    def move(self, dir):
        (dr, dc) = delta[dir]
        nr = self.robot_r + dr
        nc = self.robot_c + dc
        # check if the next cell is in bound and there is no walls linking you and no walls from the new cells
        if (not self.in_bound(nr, nc)
            or self.board[self.robot_r][self.robot_c].walls[dir]
            or self.board[nr][nc].walls[opp_dir[dir]]):
            return False
        self.robot_r = nr
        self.robot_c = nc
        return True

    def number_of_actions(self):
      return self.m*2 + self.n*2 + 4
    def number_of_input_dims(self):
      return self.n*self.m + 2
    ## A card will look like this
    ## #?#
    ## ? ?
    ## #?#
    ## ? will be X if there is a wall, else it will be empty space
    def __repr__(self):
        str_arr = [[' ' for _ in range(self.m*3)] for _ in range(self.n * 3)]
        for i in range(self.n):
            for j in range(self.m):
                for (di, dj) in [(0,0), (2,0), (0,2), (2,2)]:
                    str_arr[i*3 + di][j*3 + dj] = '#'
                for dir in range(4):
                    if self.board[i][j].walls[dir]:
                        (di, dj) = delta[dir]
                        str_arr[i*3 + 1 + di][j*3 + 1 + dj] = 'X'

        str_arr[self.robot_r * 3 + 1][self.robot_c * 3 + 1] = 'R'
        ss = ""
        for i in range(len(str_arr)):
            for j in range(len(str_arr[0])):
                ss += str_arr[i][j]
            ss += "\n"
        return ss
    def print_state(self):
        print(self)

    def get_state_repr(self):
        arr = [self.robot_r, self.robot_c]
        for i in range(self.n):
          for j in range(self.m):
            arr.append(self.board[i][j].number)
        return arr



  ## ACTIONS THAT ROBOT CAN DO (true means succeed, false means not succeed)
    def cycle_left(self, r):
        return (self.cycle_row(r, -1), "CYCLE LEFT: " + str(r))
    def cycle_right(self, r):
        return (self.cycle_row(r, 1), "CYCLE RIGHT: " + str(r))
    def cycle_down(self, c):
        return (self.cycle_col(c, 1), "CYCLE DOWN: " + str(c))
    def cycle_up(self, c):
        return (self.cycle_col(c, -1), "CYCLE UP: " + str(c))
    def move_down(self):
        return (self.move(DOWN), "MOVE DOWN")
    def move_up(self):
        return (self.move(UP), "MOVE UP")
    def move_left(self):
        return (self.move(LEFT), "MOVE LEFT")
    def move_right(self):
        return (self.move(RIGHT), "MOVE RIGHT")

      ## 0->3 are movements, 4->4 + (m-1) are cycle down, 4+m->4 + (2m-1) are cycle ups
    def do_action(self, action_no):
      if action_no == 0:
        return self.move_down()
      elif action_no == 1:
        return self.move_up()
      elif action_no == 2:
        return self.move_left()
      elif action_no == 3:
        return self.move_right()
      elif action_no < 4 + self.n:
        return self.cycle_left(action_no - 4)
      elif action_no < 4 + self.n*2:
        return self.cycle_right(action_no - 4 - self.n)
      elif action_no < 4 + self.n*2 + self.m:
        return self.cycle_down(action_no - 4 - self.n*2)
      else:
        return self.cycle_up(action_no - 4 - self.n*2 - self.m)

    def claim_victory(self):
        if self.robot_c == self.m - 1 and self.robot_r == self.n - 1 and not self.board[-1][-1].walls[DOWN]:
            return True
        return False

    # may need to reset board to start state
    def reset(self):
        self.board = [[Card(self.original_state[i][j]) for j in range(self.m)] for i in range(self.n)]
        self.robot_c = 0
        self.robot_r = 0
        self.moves_used = 0

    # update board after action taken
    def step(self, action):
        (able, info) = self.do_action(action)
        reward = -1.0
        is_done = False
        self.moves_used += 1
        if self.claim_victory():
          reward = 1000.0
          is_done = True
        else:
          #taking invalid moves waste moves...
          if not able:
            reward = -5.0
          if self.moves_used == self.max_moves:
            is_done = True
        return self.get_state_repr(), reward, is_done, info


### I COPTIED THIS FROM: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_torch_2020.py
#video: https://www.youtube.com/watch?v=wc-FxNENg9U&t=2101s

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(T.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(
                self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
                self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min




def solve_board(board: Board_CYCLEROWONLY):
  n = board.n
  m = board.m
  input_dims = board.number_of_input_dims()
  agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions= board.number_of_actions(), eps_end=0.01,
                  input_dims=[input_dims], lr=0.001)
  scores, eps_history = [], []
  n_games = 500

  for i in range(n_games):
      score = 0
      done = False
      board.reset()
      observation = board.get_state_repr()
      while not done:
          action = agent.choose_action(observation)
          observation_, reward, done, info = board.step(action)
          score += reward
          agent.store_transition(observation, action, reward,
                                  observation_, done)
          agent.learn()
          observation = observation_
      scores.append(score)
      eps_history.append(agent.epsilon)

      avg_score = np.mean(scores[-100:])

      print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

  ## WE TRAINED...
  board.reset()
  obs = board.get_state_repr()
  print(board)
  done = False
  score = 0
  agent.epsilon = 0
  while not done:
    action = agent.choose_action(observation)
    agent.epsilon = 0
    observation_, reward, done, info = board.step(action)
    score += reward
    print(info)
    print(board)
    observation = observation_


def main():
    b3 = [ [0,2,2,3], [3,0,0,2], [3,3,3,0] ]

    original_stdout = sys.stdout
    sys.stdout = open('3x4_solvable2.txt', 'w')

    print(Board(b3))
    solve_board(Board_CYCLEROWONLY(b3, max_moves=200))

    sys.stdout.close()
    sys.stdout = original_stdout

if __name__ == "__main__":
    main()
