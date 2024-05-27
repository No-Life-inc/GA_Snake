import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

class GraphMaker:
    def __init__(self, output_directory, output_file):
        self.output_directory = output_directory
        self.output_file = output_file

    def create_graph(self):
        # Get a list of all JSON files in the directory
        json_files = glob.glob('median_files/*.json')

        plt.figure(figsize=(12, 8))

        for json_file in json_files:
            # Step 1: Read the JSON file
            df = pd.read_json(json_file)

            # Step 2: Calculate the median
            median = df.median()

            # Step 3: Plot the median
            plt.plot(median, label=os.path.basename(json_file))

        plt.title('Median of Data from JSON Files')
        plt.xlabel('Columns')
        plt.ylabel('Median')
        plt.legend()

        # Set the x-axis limit
        plt.xlim(0, 50)

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)

        # Save the figure in the output directory
        plt.savefig(os.path.join(self.output_directory, self.output_file), bbox_inches='tight')

# Usage
graph_maker = GraphMaker('./omega_graph', 'combined_graph.png')
graph_maker.create_graph()