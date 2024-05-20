import random
from game import SnakeGame
from ga_brain import GABrain
import threading
import pygame
from selection_methods import top_20_percent, roulette_wheel_selection, rank_selection, tournament_selection

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, selection_method=top_20_percent):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
        self.selection_method = selection_method

    def initialize_population(self):
        # Create an initial population of random GABrain instances
        return [GABrain() for _ in range(self.population_size)]

    def evaluate_population(self):
        # Run a game for each GABrain in the population
        highest_amount_of_food_eaten = 0

        for brain in self.population:
            game = SnakeGame(brain=brain, display=False)
            snake_age, score, food_eaten = game.run()
            brain.set_fitness(snake_age, score)

            if food_eaten > highest_amount_of_food_eaten:
                highest_amount_of_food_eaten = food_eaten

        print(f"Highest amount of food eaten: {highest_amount_of_food_eaten}")

    # def selection(self):
    #     # Sort the population in descending order of fitness
    #     sorted_population = sorted(self.population, key=lambda brain: brain.fitness, reverse=True)

    #     # Select the top 20%
    #     top_20_percent = sorted_population[:int(0.2 * len(sorted_population))]

    #     # Randomly select one from the top 20%
    #     selected_brain = random.choice(top_20_percent)

    #     return selected_brain

    def selection(self):
        return self.selection_method(self.population)
    
    def crossover(self, parent1, parent2):
        # Create a new GABrain by combining the genomes of two parent GABrains
        child = GABrain()

        # Determine the crossover point (where to split the parent genomes)
        crossover_point = random.randint(0, len(parent1.genome))

        # Take the first part of the genome from parent1 and the rest from parent2
        child.genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]

        return child

    def mutation(self, child):
        # Randomly change some genes in the child's genome
        for i in range(len(child.genome)):
            if random.random() < self.mutation_rate:
                child.genome[i] = child.random_gene()
    
    def generate_new_population(self):
        # Keep the top 20% of the population
        new_population = self.population[:int(0.2 * len(self.population))]

        # Generate new brains from the top 20% until we reach the original population size
        while len(new_population) < self.population_size:
            parent1 = self.selection()
            parent2 = self.selection()
            child = self.crossover(parent1, parent2)
            self.mutation(child)
            new_population.append(child)

        self.population = new_population

    def run(self):
        game = SnakeGame(display=False)

        generation = 0  # Initialize generation outside the loop
        for generation in range(NUM_GENERATIONS):
            self.evaluate_population()

            best_brain = max(self.population, key=lambda brain: brain.fitness)
            # print(f"Generation {generation}: Best fitness = {best_brain.fitness}")

            # Run a game with the best brain of this generation
            game.brain = best_brain  # Update the brain of the Game instance
            game.display = True  # Display the game
            game.run()

            ga.generate_new_population()

        game.brain = best_brain
        game.display = True
        game.run()

if __name__ == "__main__":
    NUM_GENERATIONS = 150
    POPULATION_SIZE = 500
    MUTATION_RATE = 0.01

    ga = GeneticAlgorithm(POPULATION_SIZE, MUTATION_RATE, selection_method=top_20_percent)
    ga.run()

    pygame.quit()