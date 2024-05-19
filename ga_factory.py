import random
from game import SnakeGame
from ga_brain import GABrain
import threading

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        # Create an initial population of random GABrain instances
        return [GABrain() for _ in range(self.population_size)]

    def evaluate_population(self):
        # Run a game for each GABrain in the population
        for brain in self.population:
            game = SnakeGame(brain)
            while not game.game_over:
                game.run()  # This method should update the game state based on a fixed time step
            brain.fitness = game.score

    def selection(self):
        # Select a GABrain to reproduce based on its fitness
        total_fitness = sum(brain.fitness for brain in self.population)
        if total_fitness ==0:
            # If total fitness is zero, assign equal selection probabilities
            selection_probabilities = [1 / self.population_size for _ in self.population]
        else:
            selection_probabilities = [brain.fitness / total_fitness for brain in self.population]
        return random.choices(self.population, weights=selection_probabilities, k=1)[0]

    def crossover(self, parent1, parent2):
        # Create a new GABrain by combining the genomes of two parent GABrains
        child = GABrain()
        child.genome = [random.choice(gene_pair) for gene_pair in zip(parent1.genome, parent2.genome)]
        return child

    def mutation(self, child):
        # Randomly change some genes in the child's genome
        for i in range(len(child.genome)):
            if random.random() < self.mutation_rate:
                child.genome[i] = child.random_gene()

    def run(self):
        generation = 0  # Initialize generation outside the loop
        for generation in range(NUM_GENERATIONS):
            self.evaluate_population()
            new_population = []

            for _ in range(self.population_size):
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
    NUM_GENERATIONS = 10
    POPULATION_SIZE = 10
    MUTATION_RATE = 0.1

    ga = GeneticAlgorithm(POPULATION_SIZE, MUTATION_RATE)
    ga.run()