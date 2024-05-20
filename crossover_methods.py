import random
from ga_brain import GABrain

def single_point_crossover(parent1, parent2):
    # Create a new GABrain


    # Determine the crossover point (where to split the parent genomes)
    crossover_point = random.randint(0, len(parent1.genome))

    # Take the first part of the genome from parent1 and the rest from parent2

    child_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]

    child = GABrain(genome=child_genome)

    return child


def two_point_crossover(parent1, parent2):
    # Create a new GABrain


    # Determine the crossover points (where to split the parent genomes)
    crossover_point1 = random.randint(0, len(parent1.genome) - 1)
    crossover_point2 = random.randint(crossover_point1, len(parent1.genome))

    # Take the first part of the genome from parent1, the middle part from parent2, and the last part from parent1
    child_genome = parent1.genome[:crossover_point1] + parent2.genome[
                                                       crossover_point1:crossover_point2] + parent1.genome[
                                                                                            crossover_point2:]

    child = GABrain(genome=child_genome)
    return child


def uniform_crossover(parent1, parent2):
    # Initialize the child's genome to a list of the correct size
    child_genome = [0] * len(parent1.genome)

    # For each gene, randomly choose whether to take it from parent1 or parent2
    for i in range(len(parent1.genome)):
        child_genome[i] = parent1.genome[i] if random.random() < 0.5 else parent2.genome[i]

    child = GABrain(genome=child_genome)
    return child


def arithmetic_crossover(parent1, parent2):
    # Initialize the child's genome to a list of the correct size
    child_genome = [0] * len(parent1.genome)

    # For each gene, take the average of the corresponding genes in parent1 and parent2
    for i in range(len(parent1.genome)):
        child_genome[i] = (parent1.genome[i] + parent2.genome[i]) / 2

    child = GABrain(genome=child_genome)
    return child

