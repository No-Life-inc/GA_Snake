import pandas as pd
import glob
import os

# Get a list of all the file paths
file_paths = glob.glob('./Population_2000_data/raw_data/*/Generation_Food.csv')  # Adjust the path and file pattern as needed

# Group the file paths by combination
file_paths_by_combination = {}
for file_path in file_paths:
    folder_name = os.path.basename(os.path.dirname(file_path))  # Get the folder name
    combination, seed = '_'.join(folder_name.split('_')[:-1]), folder_name.split('_')[-1]  # Assuming the seed is the last part of the folder name
    if combination not in file_paths_by_combination:
        file_paths_by_combination[combination] = []
    file_paths_by_combination[combination].append(file_path)

# Create the directory for the combined CSV files
os.makedirs('./combined_data', exist_ok=True)

# For each combination, read the files into DataFrames, concatenate them, and write them to a new CSV file
for combination, file_paths in file_paths_by_combination.items():
    dfs = []
    for file_path in file_paths:
        if os.path.getsize(file_path) > 0:  # Check if the file is not empty
            dfs.append(pd.read_csv(file_path))
    if dfs:  # Check if the list of DataFrames is not empty
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(f'./combined_data/{combination}_combined.csv', index=False)