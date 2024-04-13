import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from bot import Board  # Assuming this imports the environment

# Constants
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# Define Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        # Define your neural network architecture
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.state_size = state_size
        self.action_size = action_size

    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(range(self.action_size))
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state).float())
                return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state).float()
        next_state_tensor = torch.tensor(next_state).float()
        q_values = self.q_network(state_tensor)
        target_q_values = q_values.clone().detach()

        if not done:
            with torch.no_grad():
                next_q_values = self.target_network(next_state_tensor)
                max_next_q_value = torch.max(next_q_values)
                target_q_values[action] = reward + self.discount_factor * max_next_q_value
        else:
            target_q_values[action] = reward

        loss = self.loss_fn(q_values, target_q_values.unsqueeze(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay

# Training function
def train():
    board = Board()
    agent = DQNAgent(state_size=board.get_state_size(), action_size=len(ACTIONS))

    for episode in range(10000):
        total_reward = 0
        state = board.reset()  # reset board to start state
        while not board.claim_victory():  # check goal state
            action = agent.choose_action(state)
            new_state, reward = board.step(action)  # update board after taking action to new state
            agent.learn(state, action, reward, new_state, board.claim_victory())
            state = new_state
            total_reward += reward

        agent.decay_exploration_rate()
        if episode % 100 == 0:
            agent.update_target_network()

        print(f"Episode: {episode}, Total reward: {total_reward}")

if __name__ == "__main__":
    train()
