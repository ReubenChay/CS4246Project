import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def read_epsilon_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Extract epsilon information from each line
    epsilons = []
    for line in lines:
        if 'epsilon' in line:
            match = re.search(r'epsilon ([\d\.-]+)', line)
            if match:
                epsilons.append(float(match.group(1)))
    return epsilons


def plot_epsilons(data_files):
    plt.figure(figsize=(12, 6))

    for filepath in data_files:
        epsilons = read_epsilon_from_file(filepath)
        episodes = list(range(1, len(epsilons) + 1))
        # Use the filename without the extension as the label
        label = os.path.splitext(os.path.basename(filepath))[0]
        plt.plot(episodes, epsilons, label=label)

    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Over Episodes for Different Boards')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # List of file paths
    data_files =  [ '5x5_unsolvable1.txt', '5x5_unsolvable2.txt']  # Add all file paths here

    # Call the function to plot the scores
    plot_epsilons(data_files)
