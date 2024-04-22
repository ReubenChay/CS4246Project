import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def read_scores_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Extract score information from each line, making sure to avoid average scores
    scores = []
    for line in lines:
        if 'score' in line:
            match = re.search(r'episode\s+\d+ score ([\d\.-]+)', line)
            if match:
                scores.append(float(match.group(1)))

    print(scores)
    return scores


def plot_scores(data_files):
    plt.figure(figsize=(12, 6))

    for filepath in data_files:
        scores = read_scores_from_file(filepath)
        episodes = list(range(1, len(scores) + 1))
        # Use the filename without the extension as the label
        label = os.path.splitext(os.path.basename(filepath))[0]
        plt.plot(episodes, scores, label=label)

    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Score Per Episode for Different Boards')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    # List of file paths
    data_files =  [ '5x5_solvable3.txt', ' 6x6_solvable.txt']  # Add all file paths here

    # Call the function to plot the scores
    plot_scores(data_files)