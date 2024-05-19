import random

import numpy as np

from game import SnakeGame
from ga_brain import GABrain
import threading

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        return [GABrain() for _ in range(self.population_size)]

    def evaluate_population(self):
        sigma_share = 5  # Adjust this parameter as needed
        for brain in self.population:
            game = SnakeGame(brain)
            while not game.game_over:
                game.run()
            brain.calculate_fitness(game)

        distances = np.zeros((self.population_size, self.population_size))
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                distances[i, j] = distances[j, i] = self.genome_distance(self.population[i], self.population[j])

        for i in range(self.population_size):
            sharing = sum(self.sharing_function(distances[i, j], sigma_share) for j in range(self.population_size))
            self.population[i].fitness /= sharing

    def genome_distance(self, brain1, brain2):
        distances = [np.linalg.norm(layer1 - layer2) for layer1, layer2 in zip(brain1.genome, brain2.genome)]
        return np.linalg.norm(distances)

    def sharing_function(self, distance, sigma_share):
        return 1 if distance < sigma_share else 0

    def selection(self):
        total_fitness = sum(brain.fitness for brain in self.population)
        selection_probabilities = [(brain.fitness + 1e-6) / (total_fitness + 1e-6) for brain in self.population]
        return random.choices(self.population, weights=selection_probabilities, k=1)[0]

    def crossover(self, parent1, parent2):
        child_genome = []
        for gene1, gene2 in zip(parent1.genome, parent2.genome):
            if np.random.rand() < 0.5:
                mask = np.random.rand(*gene1.shape) > 0.5
                new_gene = np.where(mask, gene1, gene2)
            else:
                crossover_point = np.random.randint(1, gene1.size)
                new_gene = np.concatenate((gene1.flat[:crossover_point], gene2.flat[crossover_point:])).reshape(gene1.shape)
            child_genome.append(new_gene)
        return GABrain(genome=child_genome)

    def mutation(self, child):
        for i in range(len(child.genome)):
            if np.random.rand() < self.mutation_rate:
                mutation_matrix = np.random.randn(*child.genome[i].shape) * 0.1
                child.genome[i] += mutation_matrix

    def run(self):
        generation = 0
        for generation in range(NUM_GENERATIONS):
            self.evaluate_population()
            new_population = []
            top_10_percent = int(self.population_size * 0.1)
            sorted_population = sorted(self.population, key=lambda brain: brain.fitness, reverse=True)
            new_population.extend(sorted_population[:top_10_percent])
            for _ in range(self.population_size - top_10_percent):
                parent1 = self.selection()
                parent2 = self.selection()
                child = self.crossover(parent1, parent2)
                self.mutation(child)
                new_population.append(child)
            self.population = new_population

        best_brain = max(self.population, key=lambda brain: brain.fitness)
        print(f"Generation {generation}: Best fitness = {best_brain.fitness}")

        game = SnakeGame(best_brain, display=True)
        game.game_loop()

if __name__ == "__main__":
    NUM_GENERATIONS = 100
    POPULATION_SIZE = 100
    MUTATION_RATE = 0.1

    ga = GeneticAlgorithm(POPULATION_SIZE, MUTATION_RATE)
    ga.run()
