UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
opp_dir = [DOWN, UP, RIGHT, LEFT]
delta = [(-1,0), (1,0), (0,1), (0,-1)]
#card type is denoted by a 4 bit number. 
#For example 0101 means a wall in down and a wall in right
class Card:
    def __init__(self, no):
        self.walls = [False, False, False, False]
        for idx in range(4):
            if (no&(1<<idx)) != 0:
                self.walls[idx] = True

class Board:
    def __init__(self, card_no):
        self.n = len(card_no)
        self.m = len(card_no[0])
        self.board = [[Card(card_no[i][j]) for j in range(self.m)] for i in range(self.n)]
        self.robot_r = 0
        self.robot_c = 0
    
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
        return self.cycle_row(r, 1)
    def cycle_right(self, r):
        return self.cycle_row(r, -1)
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



board_1 = [[0,0,4], [0,0,0], [0,8,8]]

random_bot(Board(board_1))

# solve from a txt file
#board = read_from_file("demo_board.txt")
#random_bot(board)

#try solve random maze, most boards are unwinnable actually
#random_bot(generate_random_board(2,2))

