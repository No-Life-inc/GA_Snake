import torch
import torch.nn.functional as F

class GABrainTorch(torch.nn.Module):
    def __init__(self, genome=None, input_nodes=24, hidden_nodes=16, output_nodes=4, hidden_layers=2):
        super(GABrainTorch, self).__init__()
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers

        self.fitness = 0  # Initialize fitness

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if genome is None:
            self.genome = self._create_genome()
        else:
            self.genome = genome

        self.weights = self._decode_genome(self.genome)

    def set_fitness(self, snake_age, game_score):
        self.fitness = snake_age + game_score
        return self.fitness

    def reset_fitness(self):
        self.fitness = 0

    def _create_genome(self):
        genome = []
        genome.append(torch.randn(self.hidden_nodes, self.input_nodes + 1).to(self.device))  # Added +1 for bias
        for _ in range(self.hidden_layers - 1):
            genome.append(torch.randn(self.hidden_nodes, self.hidden_nodes + 1).to(self.device))  # Added +1 for bias
        genome.append(torch.randn(self.output_nodes, self.hidden_nodes + 1).to(self.device))  # Added +1 for bias
        return genome

    def _decode_genome(self, genome):
        return genome

    def forward(self, inputs):
        inputs = torch.flatten(torch.tensor(inputs, device=self.device)).float()  # Convert inputs to a flat tensor
        curr_layer = torch.cat((inputs, torch.tensor([1.], device=self.device).float()))  # Append 1 for bias

        for i in range(self.hidden_layers):
            curr_layer = torch.tanh(torch.matmul(self.weights[i].float(), curr_layer))
            curr_layer = torch.cat((curr_layer, torch.tensor([1.], device=self.device).float()))  # Append 1 for bias

        output = torch.matmul(self.weights[-1].float(), curr_layer)

        direction_probabilities = F.softmax(output, dim=0)

        # Get the index of the direction with the highest probability
        direction = torch.argmax(direction_probabilities)

        return direction.item()

    def random_gene(self):
        return torch.randn(1).item()

    def mutate(self, mutation_rate):
        for i in range(len(self.weights)):
            if torch.rand(1).item() < mutation_rate:
                self.weights[i] += torch.randn(self.weights[i].shape).to(self.device) * 0.1