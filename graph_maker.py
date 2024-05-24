import matplotlib.pyplot as plt
import os
import pandas as pd
import glob

class GraphMaker:
    def __init__(self, output_directory, output_file):
        self.output_directory = output_directory
        self.output_file = output_file

    def create_graph(self):
        file_paths = glob.glob('./output/*.csv')
        plt.figure(figsize=(12, 8))

        for file_path in file_paths:
            data = pd.read_csv(file_path, index_col=0)
            data = data.head(50)  # Only include the first 50 generations
            label = os.path.basename(file_path)
            label_parts = label.split('_')  # Assuming the parts of the label are separated by underscores
            short_label = '_'.join(label_parts[:5])  # Only keep the first five parts
            plt.plot(data, label=short_label)

        plt.xlabel('Generation')
        plt.ylabel('Median Food')
        
        # Add the new label
        plt.text(0.10, 0.98, '- Random Mutation\n- 2000 Population\n- 0.05 Mutation Rate\n- 0.1 Elitism', transform=plt.gcf().transFigure, ha='left', va='top')

        # Add the group label
        plt.text(0.80, 0.95, 'Group 2', transform=plt.gcf().transFigure, ha='left', va='top')
        
        # Adjusting the legend position to be above the plot
        plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2, fontsize='small')
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Save the figure in the output directory
        plt.savefig(os.path.join(self.output_directory, self.output_file), bbox_inches='tight')

# Usage
graph_maker = GraphMaker('./omega_graph', 'combined_graph.png')
graph_maker.create_graph()