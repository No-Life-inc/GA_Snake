import torch
from pytorch.pytorch_ga_brain import GABrainTorch

def single_point_crossover(parent1, parent2):
    crossover_point = torch.randint(0, len(parent1.genome), (1,)).item()
    child_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
    child = GABrainTorch(genome=child_genome)
    return child

def two_point_crossover(parent1, parent2):
    crossover_point1 = torch.randint(0, len(parent1.genome) - 1, (1,)).item()
    crossover_point2 = torch.randint(crossover_point1, len(parent1.genome), (1,)).item()
    child_genome = parent1.genome[:crossover_point1] + parent2.genome[crossover_point1:crossover_point2] + parent1.genome[crossover_point2:]
    child = GABrainTorch(genome=child_genome)
    return child

def uniform_crossover(parent1, parent2):
    child_genome = [parent1.genome[i] if torch.rand(1).item() < 0.5 else parent2.genome[i] for i in range(len(parent1.genome))]
    child = GABrainTorch(genome=child_genome)
    return child
