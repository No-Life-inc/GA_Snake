import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

class GABrain:
    def __init__(self, genome=None, input_nodes=25, hidden_nodes=25, output_nodes=4, hidden_layers=2):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers

        self.fitness = 0  # Initialize fitness

        if genome is None:
            self.genome = self._create_genome()
        else:
            self.genome = genome

        self.weights = self._decode_genome(self.genome)

    def set_fitness(self, snake_age, game_score):
        self.fitness = snake_age * game_score

        return self.fitness

    def _create_genome(self):
        genome = []
        genome.append(np.random.randn(self.hidden_nodes, self.input_nodes + 1))  # +1 for bias
        for _ in range(self.hidden_layers - 1):
            genome.append(np.random.randn(self.hidden_nodes, self.hidden_nodes + 1))  # +1 for bias
        genome.append(np.random.randn(self.output_nodes, self.hidden_nodes + 1))  # +1 for bias
        return genome

    def _decode_genome(self, genome):
        return genome

    def forward(self, inputs):
        inputs = np.ravel(np.array(inputs))  # Convert inputs to a flat numpy array
        inputs = np.append(inputs, 1)  # Add bias to the inputs
        curr_layer = inputs

        for i in range(self.hidden_layers):
            curr_layer = tanh(np.dot(self.weights[i], curr_layer))
            curr_layer = np.append(curr_layer, 1)  # Add bias to the output of the hidden layer

        output = sigmoid(np.dot(self.weights[-1], curr_layer))
        direction_probabilities = softmax(output)

        # Get the index of the direction with the highest probability
        direction = np.argmax(direction_probabilities)

        return direction

    def random_gene(self):
        return np.random.randn()

    def mutate(self, mutation_rate):
        for i in range(len(self.weights)):
            if np.random.rand() < mutation_rate:
                self.weights[i] += np.random.randn(*self.weights[i].shape) * 0.1