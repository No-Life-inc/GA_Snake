import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SnakeBrain:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, hidden_layers):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers

        self.weights = [np.random.randn(self.hidden_nodes, self.input_nodes + 1)]
        for _ in range(self.hidden_layers - 1):
            self.weights.append(np.random.randn(self.hidden_nodes, self.hidden_nodes + 1))
        self.weights.append(np.random.randn(self.output_nodes, self.hidden_nodes + 1))

    def forward(self, inputs):
        inputs = np.append(inputs, 1)  # Add bias to the inputs
        curr_layer = inputs

        for i in range(self.hidden_layers):
            curr_layer = sigmoid(np.dot(self.weights[i], curr_layer))
            curr_layer = np.append(curr_layer, 1)  # Add bias to the output of the hidden layer

        output = sigmoid(np.dot(self.weights[-1], curr_layer))

        return output
