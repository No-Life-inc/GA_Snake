import pandas as pd
import os
from collections import defaultdict
import json

# Define the directory where the CSV files are located
directory = './raw_data/combined'

# Define the combinations
combinations = ['alpha_selection_single_point_crossover', 'alpha_selection_two_point_crossover', 
                'rank_selection_single_point_crossover', 'rank_selection_two_point_crossover', 
                'truncation_selection_single_point_crossover', 'truncation_selection_two_point_crossover', 
                'tournament_selection_single_point_crossover', 'tournament_selection_two_point_crossover']

# Create a defaultdict for each combination
results = {combination: defaultdict(list) for combination in combinations}

# Iterate over all subdirectories in the directory
for subdir, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a CSV file
        if file.endswith('Generation_Food.csv'):
            # Construct the full file path
            file_path = os.path.join(subdir, file)
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Get the subdirectory name
                subdir_name = os.path.basename(subdir)
                
                # Check if the subdirectory name starts with any of the combinations
                for combination in combinations:
                    if subdir_name.startswith(combination):
                        # Append the 'Food' values to the corresponding generation in the result dict
                        for index, row in df.iterrows():
                            if int(row['Generation']) <= 50:
                                results[combination][int(row['Generation'])].append(int(row['Food']))
                        break
            except pd.errors.EmptyDataError:
                print(f"Error reading file in subdirectory: {subdir} - {file}")

# Convert the defaultdicts to regular dicts and save them to JSON files
for combination, result in results.items():
    result_dict = dict(result)
    with open(f'{combination}_result.json', 'w') as f:
        json.dump(result_dict, f)
