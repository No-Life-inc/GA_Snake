import random
import numpy as np
from game import SnakeGame
from ga_brain import GABrain
import pygame
import matplotlib.pyplot as plt
import os

from selection_methods import top_20_percent, roulette_wheel_selection, rank_selection, tournament_selection, elitism_selection, alpha_selection
from crossover_methods import single_point_crossover, two_point_crossover, uniform_crossover, arithmetic_crossover


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, number_of_generations ,selection_method=tournament_selection,
                 crossover_methods=single_point_crossover, elitism_rate=0.0, display_best_snake=False):

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.number_of_generations = number_of_generations
        self.population = self.initialize_population()
        self.gen_best_score_dict = {}
        self.gen_avg_fitness_dict = {}
        self.gen_best_fitness_dict = {}
        self.selection_method = selection_method
        self.display_best_snake = display_best_snake
        self.crossover_method = crossover_methods
        self.elitism_rate = elitism_rate
        self.path = self.make_subdirs()


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

        # print(f"Generation {generation_number} - Highest amount of food eaten: {highest_amount_of_food_eaten}")

        if best_game is not None:
            best_game.save_game_states(f'best_snakes/{self.path}/Gen_{generation_number}_snake.pkl')

            if self.display_best_snake:
                best_game.play_back(f'best_snakes/{self.path}/Gen_{generation_number}_snake.pkl')

        gen_fitness = [brain.fitness for brain in self.population]
        self.gen_avg_fitness_dict[generation_number] = sum(gen_fitness) / len(gen_fitness)
        self.gen_best_score_dict[generation_number] = highest_amount_of_food_eaten
        self.gen_best_fitness_dict[generation_number] = max(gen_fitness)

    def crossover(self, parent1, parent2):
        return self.crossover_method(parent1, parent2)

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

        elites = elitism_selection(self.population, self.elitism_rate)
        new_population = elites[:]

        while len(new_population) < self.population_size:
            selected = self.selection()

            if isinstance(selected, (list, tuple)) and len(selected) == 2:
                parent1, parent2 = selected
            else:
                parent1 = selected
                parent2 = selected
                while parent2 == parent1:
                    parent2 = self.selection()

            child = self.crossover(parent1, parent2)
            self.mutation(child)
            new_population.append(child)

        self.population = new_population
        for brain in self.population:
            brain.reset_fitness()

    def run(self):
        for generation in range(self.number_of_generations):
            self.evaluate_population(generation)
            best_brain = max(self.population, key=lambda brain: brain.fitness)
            self.generate_new_population()

        plt = self.make_plot()
        self.save_plot(plt)

        self.save_score_data()
        self.save_fitness_data()


    def make_plot(self):
        keys = list(self.gen_best_score_dict.keys())
        values = [int(value) for value in self.gen_best_score_dict.values()]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)  # Create an Axes object

        ax.scatter(keys, values)
        ax.plot(keys, values)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Score')
        ax.set_title('Generation vs Score')
        ax.set_yticks(range(0, max(values) + 1, 1))
        ax.set_xticks(range(1, max(keys) + 1, 1))

        return fig
    
    def save_plot(self, plt):

        # Create the filename
        filename = f'graphs/{self.path}/Generation_vs_score_plot.png'

        # Save the plot
        plt.savefig(filename)

    #make subdirs for best_snakes and graphs and raw data with selection, crossover method, population size, mutation rate
    def make_subdirs(self):
        selection_method_name = self.selection_method.__name__
        crossover_method_name = self.crossover_method.__name__

        subdir_name = f'{selection_method_name}_{crossover_method_name}_{self.population_size}_{self.mutation_rate}_elitism_{self.elitism_rate}'

        for parent_dir in ['best_snakes', 'graphs', 'raw_data']:
            path = f'{parent_dir}/{subdir_name}'
            os.makedirs(path, exist_ok=True)

        return subdir_name
    
    def save_score_data(self):
        # Save gen_score_dict to a csv file under path

        filename = f'raw_data/{self.path}/Generation_score.csv'

        with open(filename, 'w') as f:
            f.write('Generation,Score\n')
            for generation, score in self.gen_best_score_dict.items():
                f.write(f'{generation},{score}\n')

    def save_fitness_data(self):
        filename = f'raw_data/{self.path}/Generation_fitness.csv'

        with open(filename, 'w') as f:
            f.write('Generation,Best Fitness,Avg Fitness\n')
            for generation in self.gen_best_fitness_dict:
                f.write(f'{generation},{self.gen_best_fitness_dict[generation]},{self.gen_avg_fitness_dict[generation]}\n')
