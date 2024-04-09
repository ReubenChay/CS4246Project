import numpy as np
import random
from bot import Board

# Constants
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]



class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = {}  # nested dictionary that maps (state, action) -> q_value
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def learn(self, old_state, action, reward, new_state):
        # Convert states to a string to use as dictionary keys
        old_state_key = str(old_state)
        new_state_key = str(new_state)

        # Initialize Q-values to 0 if they're not already in the Q-table
        if old_state_key not in self.q_table:
            self.q_table[old_state_key] = {a: 0 for a in self.actions}
        if new_state_key not in self.q_table:
            self.q_table[new_state_key] = {a: 0 for a in self.actions}

        # Update the Q-value for the old state and action pair
        max_q_new_state = max(self.q_table[new_state_key].values())
        self.q_table[old_state_key][action] = self.q_table[old_state_key][action] + self.learning_rate * (reward + self.discount_factor * max_q_new_state - self.q_table[old_state_key][action])

    def choose_action(self, state):
        state_key = str(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0 for a in self.actions}

        # Exploration vs Exploitation
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(self.actions)
        else:
            action = max(self.q_table[state_key], key=self.q_table[state_key].get)

        return action


# functions to convert board states to and from string keys
def state_to_key(state):
    pass

def key_to_state(key):
    pass


def train():
    board = Board()
    agent = QLearningAgent(ACTIONS)

    for episode in range(10000):
        total_reward = 0
        state = board.reset()  # reset board to start state
        while not board.claim_victory():  # check goal state
            action = agent.choose_action(state)
            new_state, reward = board.step(action)  # update board after taking action to new state
            agent.learn(state, action, reward, new_state)
            state = new_state
            total_reward += reward

        print(f"Episode: {episode}, Total reward: {total_reward}")


if __name__ == "__main__":
    train()
