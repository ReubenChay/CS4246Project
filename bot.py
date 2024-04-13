UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
opp_dir = [DOWN, UP, LEFT, RIGHT]
delta = [(-1,0), (1,0), (0,1), (0,-1)]
#Card type is denoted by a 4 bit number. For example: 0101 means a wall in down and a wall in left
class Card:
    def __init__(self, no):
        self.walls = [False, False, False, False]
        for idx in range(4):
            if (no&(1<<idx)) != 0:
                self.walls[idx] = True

    def walls_to_number(self):
        # Convert wall configuration back to a number representation for cloning
        num = 0
        for idx in range(4):
            if self.walls[idx]:
                num |= 1 << idx
        return num

class Board:
    def __init__(self, card_no):
        self.n = len(card_no)
        self.m = len(card_no[0])
        self.board = [[Card(card_no[i][j]) for j in range(self.m)] for i in range(self.n)]
        self.robot_r = 0
        self.robot_c = 0
    
    def clone(self):
        # Create a deep copy of the board to ensure simulations do not interfere with actual game state
        cloned_card_no = [[self.board[i][j].walls_to_number() for j in range(self.m)] for i in range(self.n)]
        return Board(cloned_card_no)
    
    def in_bound(self, r, c):
        return r in range(0, self.n) and c in range(0, self.m)

    def cycle_col(self, c, offset):
        if self.robot_c == c:
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

## ACTIONS THAT ROBOT CAN DO (true means succeed, false means not succeed)
    def cycle_left(self, r):
        return self.cycle_row(r, -1)
    def cycle_right(self, r):
        return self.cycle_row(r, 1)
    def cycle_down(self, c):
        return self.cycle_col(c, 1)
    def cycle_up(self, c):
        return self.cycle_col(c, -1)
    def move_down(self):    
        return self.move(DOWN)
    def move_up(self):
        return self.move(UP)
    def move_left(self):
        return self.move(LEFT)
    def move_right(self):
        return self.move(RIGHT)

    def claim_victory(self):
        if self.robot_c == self.m - 1 and self.robot_r == self.n - 1 and not self.board[-1][-1].walls[DOWN]:
            return True
        return False
    
import random
def generate_random_board(n, m):
    return Board([[random.randint(0, 15) for _ in range(m)] for _ in range(n)])

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
    return Board(arr)



# bot randomly move in 4 directions if possible
def random_bot(board: Board):
    print(board)
    MAX = 100
    completed = False
    moves_used = 0
    for _ in range(0, MAX):
        if board.claim_victory():
            print("WE WIN!")
            completed = True
            break
        success = False
        dd = random.randint(0,3)
        if dd == 0:
            if board.move_up():
                success = True
                print("MOVING UP")
        elif dd == 1:
            if board.move_down():
                success = True
                print("MOVING DOWN")
        elif dd == 2:
            if board.move_left():
                success = True
                print("MOVING LEFT")
        else:
            if board.move_right():
                success = True
                print("MOVE RIGHT")
        if success:
            moves_used += 1
            print("Current board after %d moves: " %moves_used)
            print(board)
    if not completed:
        print("I GIVE UP!")



#board_1 = [[0,0,4], [0,0,0], [0,8,8]]

#random_bot(Board(board_1))

# solve from a txt file
#board = read_from_file("demo_board.txt")
#random_bot(board)

#try solve random maze, most boards are unwinnable actually
#random_bot(generate_random_board(2,2))

import math
from collections import defaultdict

class MCTSNode:
    def __init__(self, board, parent=None, action=None):
        self.board = board
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = self.get_possible_actions()
        self.is_terminal = board.claim_victory()

    def get_possible_actions(self):
        actions = []
        # Avoid adding the opposite of the last action unless no other options are available
        last_action = self.parent.action if self.parent else None

        # Movement actions with reversal prevention
        if self.board.robot_r > 0 and not self.board.board[self.board.robot_r][self.board.robot_c].walls[UP] and last_action != ("move", DOWN):
            actions.append(("move", UP))
        if self.board.robot_r < self.board.n - 1 and not self.board.board[self.board.robot_r][self.board.robot_c].walls[DOWN] and last_action != ("move", UP):
            actions.append(("move", DOWN))
        if self.board.robot_c > 0 and not self.board.board[self.board.robot_r][self.board.robot_c].walls[LEFT] and last_action != ("move", RIGHT):
            actions.append(("move", LEFT))
        if self.board.robot_c < self.board.m - 1 and not self.board.board[self.board.robot_r][self.board.robot_c].walls[RIGHT] and last_action != ("move", LEFT):
            actions.append(("move", RIGHT))
        
        # Add cycling actions with checks for immediate reversals
        for i in range(self.board.n):
            if i != self.board.robot_r:
                actions.append(("cycle_row", i, 1))
                actions.append(("cycle_row", i, -1))
        for j in range(self.board.m):
            if j != self.board.robot_c:
                actions.append(("cycle_col", j, 1))
                actions.append(("cycle_col", j, -1))
        return actions

    def select_child(self):
        # Select child with highest UCT value
        c = 2  # exploration constant
        log_visits = math.log(self.visits)
        return max(self.children, key=lambda child: child.wins / child.visits + c * math.sqrt(log_visits / child.visits))

    def expand(self):
        action = self.untried_actions.pop(0)
        new_board = self.board.clone()
        if action[0] == "move":
            new_board.move(action[1])
        elif action[0] == "cycle_row":
            new_board.cycle_row(action[1], action[2])
        elif action[0] == "cycle_col":
            new_board.cycle_col(action[1], action[2])
        child_node = MCTSNode(new_board, self, action)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1 - result)  # Propagate the opposite result up

def monte_carlo_tree_search(root, simulations=100):
    for _ in range(simulations):
        node = root
        # Select the best child node until we find an unexpanded node
        while node.untried_actions == [] and node.children != []:
            node = node.select_child()
        
        # If there are untried actions, expand the node
        if node.untried_actions != []:
            node = node.expand()
        
        # Perform a random simulation from the new node
        outcome = simulate_random_game(node)
        
        # Backpropagate the result of the simulation
        node.backpropagate(outcome)
    
    # Return the action from the root's children with the highest number of visits
    return max(root.children, key=lambda c: c.visits).action if root.children else None

def simulate_random_game(node):
    step_counter = 0  # Add a counter to prevent infinite loops
    while not node.is_terminal and step_counter < 100:  # Limit steps to prevent infinite loops
        possible_moves = node.get_possible_actions()
        if not possible_moves:
            #print("No more possible moves from this node.")
            return 0  # Lose if no moves are possible
        action = random.choice(possible_moves)
        #print(f"Attempting action: {action}")
        node = node.expand()  # Assume expand handles simulation from action
        step_counter += 1
        node.is_terminal = node.board.claim_victory()  # Update terminal status
    return 1  # Win if terminal condition is met


def run_mcts(board):
    current_board = board
    steps = 0
    while not current_board.claim_victory() and steps < 100:  # Limit to prevent infinite loop
        #print("Current board state:")
        print(current_board)
        root = MCTSNode(current_board)
        best_action = monte_carlo_tree_search(root, 50)  # Reduced for faster debug cycles
        if best_action:
            print(f"Best action determined by MCTS: {best_action}")
            action_type, *params = best_action
            execute_action(current_board, action_type, params)
            steps += 1
        else:
            print("No possible actions found or all simulations led to terminal nodes.")
            break

        print(f"Board after action {steps}:")
        #print(current_board)

    if current_board.claim_victory():
        print("Victory achieved!")
        print(f"Total steps taken: {steps}")
    else:
        print("Game ended without victory.")
        print(current_board)

def execute_action(board, action_type, params):
    if action_type == "move":
        if board.move(params[0]):
            print(f"Moved {['UP', 'DOWN', 'RIGHT', 'LEFT'][params[0]]}")
        else:
            print("Move failed")
    elif action_type == "cycle_row":
        if board.cycle_row(params[0], params[1]):
            print(f"Cycled row {params[0]} {'left' if params[1] == 1 else 'right'}")
        else:
            print("Cycle row failed")
    elif action_type == "cycle_col":
        if board.cycle_col(params[0], params[1]):
            print(f"Cycled column {params[0]} {'down' if params[1] == 1 else 'up'}")
        else:
            print("Cycle column failed")


# Example usage:
board_1 = [[0,0,4], [0,0,0], [0,8,8]]
run_mcts(Board(board_1))
