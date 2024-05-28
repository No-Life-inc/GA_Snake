import pandas as pd
import glob
import os

class FoodMedianCalculator:
    def __init__(self, directory, output_directory):
        self.directory = directory
        self.output_directory = output_directory

    def calculate(self):
        file_paths = glob.glob('./combined_data/pop500-mut01/*.csv')
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)
        
        for file_path in file_paths:
            all_data = pd.read_csv(file_path)
            median_food = all_data.groupby('Generation')['Food'].median()
            
            # Get the base name of the file and remove the extension
            base_name = os.path.basename(file_path)
            file_name_without_extension = os.path.splitext(base_name)[0]
            
            # Write the median food values to a CSV file in the output directory
            median_food.to_csv(os.path.join(self.output_directory, f'{file_name_without_extension}_median.csv'), index=True, header=True)

# Usage
calculator = FoodMedianCalculator('./combined_data', './output')
calculator.calculate()