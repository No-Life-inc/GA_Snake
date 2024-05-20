import random
import numpy as np
from game import SnakeGame
from ga_brain import GABrain
import pygame
import matplotlib.pyplot as plt
from selection_methods import top_20_percent, roulette_wheel_selection, rank_selection, tournament_selection

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, selection_method=tournament_selection):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
        self.gen_score_dict = {}
        self.selection_method = selection_method

    def initialize_population(self):
        # Create an initial population of random GABrain instances
        return [GABrain() for _ in range(self.population_size)]

    def evaluate_population(self, generation_number: int):
        # Run a game for each GABrain in the population
        highest_amount_of_food_eaten = 0
        best_game = None
        generation_number += 1

        for brain in self.population:
            game = SnakeGame(brain=brain, display=False)
            snake_age, score, food_eaten = game.run()
            brain.set_fitness(snake_age, score)

            if food_eaten > highest_amount_of_food_eaten:
                highest_amount_of_food_eaten = food_eaten
                best_game = game

        print(f"Generation {generation_number} - Highest amount of food eaten: {highest_amount_of_food_eaten}")

        if best_game is not None:
            best_game.save_game_states(f'best_snakes/Gen_{generation_number}_snake.pkl')
            best_game.play_back(f'best_snakes/Gen_{generation_number}_snake.pkl')

        self.gen_score_dict[generation_number] = highest_amount_of_food_eaten

    def crossover(self, parent1, parent2):
        # Create a new GABrain by combining the genomes of two parent GABrains
        crossover_point = random.randint(0, len(parent1.genome))
        child_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
        child = GABrain(genome=child_genome)
        return child

    def mutation(self, child):
        # Randomly change some genes in the child's genome
        for i, layer in enumerate(child.genome):
            mutation_mask = np.random.rand(*layer.shape) < self.mutation_rate
            random_genes = np.random.randn(*layer.shape)  # Assuming normal distribution for random genes
            child.genome[i] = np.where(mutation_mask, random_genes, layer)

    def selection(self):
        return self.selection_method(self.population)

    def generate_new_population(self):
        self.population.sort(key=lambda brain: brain.fitness, reverse=True)

        # Determine the number of individuals to keep
        num_elites = int(self.population_size * ELITISM_RATE)

        # Keep the best individuals
        new_population = self.population[:num_elites]

        # Print the top 20 best of the population on one line in an array
        print([brain.fitness for brain in new_population[:20]])

        # Alpha crossover strategy
        alpha = self.population[0]  # Best individual
        mothers = self.population[1:num_elites]  # Remaining top individuals

        # Generate new brains until we reach the original population size
        while len(new_population) < self.population_size:
            mother = random.choice(mothers)
            child = self.crossover(alpha, mother)
            self.mutation(child)
            new_population.append(child)

        self.population = new_population
        for brain in self.population:
            brain.reset_fitness()

    def run(self):
        for generation in range(NUM_GENERATIONS):
            self.evaluate_population(generation)
            best_brain = max(self.population, key=lambda brain: brain.fitness)
            print(f"Generation {generation}: Best fitness = {best_brain.fitness}")
            self.generate_new_population()

    def make_plot(self):
        keys = list(self.gen_score_dict.keys())
        values = [int(value) for value in self.gen_score_dict.values()]
        plt.scatter(keys, values)
        plt.plot(keys, values)
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.title('Generation vs Score')
        plt.yticks(range(0, max(values)+1, 1))
        plt.xticks(range(1, max(keys)+1, 1))
        plt.show()

if __name__ == "__main__":
    NUM_GENERATIONS = 50
    POPULATION_SIZE = 2000
    MUTATION_RATE = 0.05
    ELITISM_RATE = 0.1

    ga = GeneticAlgorithm(POPULATION_SIZE, MUTATION_RATE, selection_method=tournament_selection)
    ga.run()
    ga.make_plot()

    pygame.quit()
