import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class GABrain:
    def __init__(self, genome=None, input_nodes=24, hidden_nodes=16, output_nodes=4, hidden_layers=2):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers

        fitness = 0

        if genome is None:
            self.genome = self._create_genome()
        else:
            self.genome = genome

        self.weights = self._decode_genome(self.genome)

    def _create_genome(self):
        genome = []
        genome.append(np.random.randn(self.hidden_nodes, self.input_nodes + 1))
        for _ in range(self.hidden_layers - 1):
            genome.append(np.random.randn(self.hidden_nodes, self.hidden_nodes + 1))
        genome.append(np.random.randn(self.output_nodes, self.hidden_nodes + 1))
        return genome

    def _decode_genome(self, genome):
        return genome

    def forward(self, inputs):
        inputs = np.append(inputs, 1)  # Add bias to the inputs
        curr_layer = inputs

        for i in range(self.hidden_layers):
            curr_layer = sigmoid(np.dot(self.weights[i], curr_layer))
            curr_layer = np.append(curr_layer, 1)  # Add bias to the output of the hidden layer

        output = sigmoid(np.dot(self.weights[-1], curr_layer))
        direction = np.argmax(output)

        return direction
