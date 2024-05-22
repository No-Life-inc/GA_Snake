#make plot with best score for each combination
# csvs are saved in raw_data folder under subfolders with names for the combinations
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

    # #make a dict with the name of the subfolders, that contains the generation number and the best score
best_scores = {}

    # window_size = 10  # Size of the rolling window

    # for subdir, dirs, files in os.walk('raw_data'):
    #     for file in files:
    #         if file == 'Generation_score.csv':  # Look for 'Generation_fitness.csv' specifically
    #             df = pd.read_csv(os.path.join(subdir, file))
    #             print(f"Columns in {file}: {df.columns}")  # Print column names
    #             df['Score'] = df['Score'].rolling(window=window_size).mean()  # Apply rolling mean
    #             best_scores[subdir] = df['Score']  # Replace 'Score' with 'Best Fitness'

    # #make one plot with all the best scores
    # plt.figure(figsize=(10, 5))
    # for key, value in best_scores.items():
    #     plt.plot(value, label=key)
    # plt.xlabel('Generation')
    # plt.ylabel('Best score')
    # plt.legend()
    # plt.show()
    
window_size = 10  # Size of the rolling window

    # for subdir, dirs, files in os.walk('raw_data'):
    #     for file in files:
    #         if file == 'Generation_score.csv':  # Look for 'Generation_fitness.csv' specifically
    #             df = pd.read_csv(os.path.join(subdir, file))
    #             print(f"Columns in {file}: {df.columns}")  # Print column names
    #             df['Score Increase (%)'] = df['Score'].pct_change() * 100  # Calculate percentage increase
    #             df['Score Increase (%)'] = df['Score Increase (%)'].rolling(window=window_size).mean()  # Apply rolling mean
    #             best_scores[subdir] = df['Score Increase (%)']  # Replace 'Score' with 'Score Increase (%)'

    # #make one plot with all the best scores
    # plt.figure(figsize=(10, 5))
    # for key, value in best_scores.items():
    #     plt.plot(value, label=key)
    # plt.xlabel('Generation')
    # plt.ylabel('Score Increase (%)')  # Change y label to 'Score Increase (%)'
    # plt.legend()
    # plt.show()



median_scores = []  # List to store the median scores
subdirs = []  # List to store the subdirectories

colors = cm.rainbow(np.linspace(0, 1, 10))  # Get 10 different colors

for subdir, dirs, files in os.walk('raw_data'):
    for file in files:
        if file == 'Generation_score.csv':  # Look for 'Generation_fitness.csv' specifically
            df = pd.read_csv(os.path.join(subdir, file))
            median_scores.append(df['Score'].median())  # Get the median score
            subdir_name = os.path.basename(subdir)  # Get the last part of the path
            subdir_name = subdir_name.split('_2000')[0]  # Split the string at '_2000' and take the first part
            subdirs.append(subdir_name)  # Add this line

# Create a bar chart with all the median scores
plt.figure(figsize=(10, 5))
plt.bar(subdirs, median_scores, color=colors)
plt.xlabel('Subdirectory')
plt.ylabel('Median Score')
plt.xticks(rotation='vertical')
plt.show()