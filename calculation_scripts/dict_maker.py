import pandas as pd
import os
from collections import defaultdict
import json

# Define the directory where the CSV files are located
directory = 'raw_data/mutation_rate_0.1/raw_data_pop_3000'

# Create a defaultdict to store the results
result = defaultdict(list)

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

                # Append the 'Food' values to the corresponding generation in the result dict
                for index, row in df.iterrows():
                    result[int(row['Generation'])].append(int(row['Food']))
            except pd.errors.EmptyDataError:
                print(f"Error reading file in subdirectory: {subdir}")

# Convert the defaultdict to a regular dict
result_dict = dict(result)

# Save the dictionary to a JSON file
with open('median_files/mut0.1_pop3000.json', 'w') as f:
    json.dump(result_dict, f)
