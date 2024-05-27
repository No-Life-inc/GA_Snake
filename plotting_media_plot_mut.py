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
        json_files = glob.glob('crossover_selections/*.json')

        plt.figure(figsize=(12, 8))

        

        for json_file in json_files:
            # Step 1: Read the JSON file
            df = pd.read_json(json_file)
            label = os.path.basename(json_file)
            label_parts = label.split('_')  # Assuming the parts of the label are separated by underscores
            short_label = '_'.join(label_parts[:5])  # Only keep the first five parts
            # Step 2: Calculate the median
            median = df.median()

            # Step 3: Plot the median
            plt.plot(median, label=short_label)

        plt.title('Performance of crossovers/selections across parameter combinations', y=1.15)
        plt.xlabel('Generations')
        plt.ylabel('Median')
        plt.legend()

        plt.text(0.10, 0.98, '- Random Mutation\n- All Population\n- 0.05 & 0.1 Mutation Rate\n- 0.1 Elitism', transform=plt.gcf().transFigure, ha='left', va='top')
        plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2, fontsize='small')

        # Set the x-axis limit
        plt.xlim(0, 50)

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)

        # Save the figure in the output directory
        plt.savefig(os.path.join(self.output_directory, self.output_file), bbox_inches='tight')

# Usage
graph_maker = GraphMaker('./omega_graph', 'combined_graph.png')
graph_maker.create_graph()