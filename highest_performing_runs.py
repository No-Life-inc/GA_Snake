import pandas as pd
import os
from collections import defaultdict
import json
import numpy as np

# Define the directory where the CSV files are located
directory = 'groups'

# Define the combinations
combinations = ['alpha_selection_single_point_crossover', 'alpha_selection_two_point_crossover', 
                'rank_selection_single_point_crossover', 'rank_selection_two_point_crossover', 
                'truncation_selection_single_point_crossover', 'truncation_selection_two_point_crossover', 
                'tournament_selection_single_point_crossover', 'tournament_selection_two_point_crossover']

# Create a defaultdict for each combination in each group
results = {group: {combination: defaultdict(list) for combination in combinations} for group in range(1, 7)}

# Create a dict to store the highest median 'Food' value for each combination in each group
max_median_food = {group: {combination: 0 for combination in combinations} for group in range(1, 7)}

# Create a dict to store the highest 'Food' value for each combination in each group
max_food = {group: {combination: 0 for combination in combinations} for group in range(1, 7)}

# Iterate over all subdirectories in the directory
for subdir, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a CSV file
        if file.endswith('Generation_Food.csv'):
            # Construct the full file path
            file_path = os.path.join(subdir, file)
            
            try:
                # Check if the file is empty
                if os.stat(file_path).st_size == 0:
                    print(f"File is empty: {file_path}")
                    continue

                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Get the group number and subdirectory name
                group = int(os.path.basename(os.path.dirname(subdir)))
                subdir_name = os.path.basename(subdir)
                
                # Check if the subdirectory name starts with any of the combinations
                for combination in combinations:
                    if subdir_name.startswith(combination):
                        # Append the 'Food' values to the corresponding generation in the result dict
                        for index, row in df.iterrows():
                            generation = int(row['Generation'])
                            if generation > 50:
                                break
                            food = int(row['Food'])
                            results[group][combination][int(row['Generation'])].append(food)
                            max_food[group][combination] = max(max_food[group][combination], food)
                        break

            except pd.errors.EmptyDataError:
                print(f"Error reading file in subdirectory: {subdir}")

# Calculate the median 'Food' value for each generation in each combination and update the highest median 'Food' value
for group, result in results.items():
    for combination, values in result.items():
        for generation, food_values in values.items():
            median_food = np.median(food_values)
            max_median_food[group][combination] = max(max_median_food[group][combination], median_food)

# Save the highest median 'Food' values and the highest 'Food' values to a JSON file
with open('max_median_and_max_food.json', 'w') as f:
    json.dump({'max_median_food': max_median_food, 'max_food': max_food}, f)