from game import SnakeGame
from pytorch.pytorch_ga_brain import GABrainTorch  # Import the PyTorch version of GABrain
import matplotlib.pyplot as plt
import os
import torch
from multiprocessing import Pool

from selection_methods import top_20_percent, roulette_wheel_selection, rank_selection, tournament_selection, elitism_selection, alpha_selection
from crossover_methods import single_point_crossover, two_point_crossover, uniform_crossover, arithmetic_crossover


class GeneticAlgorithmTorch:
    def __init__(self, population_size, mutation_rate, number_of_generations ,selection_method=tournament_selection,
                 crossover_methods=single_point_crossover, elitism_rate=0.0, display_best_snake=False, seed=None):

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if seed is not None:
            self.seed = seed

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
        return [GABrainTorch() for _ in range(self.population_size)]

    def evaluate_population(self, generation_number: int):
        # Run a game for each GABrain in the population
        highest_amount_of_food_eaten = 0
        generation_number += 1

        # Create a pool of worker processes
        with Pool() as pool:
            # Use the map function to run the game for each brain in the population in parallel
            results = pool.map(self.run_game, self.population)

        for brain, (game, snake_age, score, food_eaten) in zip(self.population, results):
            brain.set_fitness(snake_age, score)
        
        # Find the best brain in the population
        best_brain_index = max(range(len(self.population)), key=lambda index: self.population[index].fitness)
        best_brain = self.population[best_brain_index]

        # Get the game associated with the best brain
        best_game, _, _, highest_amount_of_food_eaten = results[best_brain_index]

        if best_game is not None:
            best_game.save_game_states(f'best_snakes/{self.path}/Gen_{generation_number}_snake.pkl')

            if self.display_best_snake:
                best_game.play_back(f'best_snakes/{self.path}/Gen_{generation_number}_snake.pkl')

        gen_fitness = [brain.fitness for brain in self.population]
        self.gen_avg_fitness_dict[generation_number] = sum(gen_fitness) / len(gen_fitness)
        self.gen_best_score_dict[generation_number] = highest_amount_of_food_eaten
        self.gen_best_fitness_dict[generation_number] = max(gen_fitness)

        return best_brain

    def run_game(self, brain):
        game = SnakeGame(brain=brain, display=False)
        snake_age, score, food_eaten = game.run()
        return game, snake_age, score, food_eaten

    def crossover(self, parent1, parent2):
        return self.crossover_method(parent1, parent2)

    def mutation(self, child):
    # Randomly change some genes in the child's genome
        for i, layer in enumerate(child.genome):
            mutation_mask = torch.rand(*layer.shape, device=child.device) < self.mutation_rate
            random_genes = torch.randn(*layer.shape, device=child.device)  # Assuming normal distribution for random genes
            child.genome[i] = torch.where(mutation_mask, random_genes, layer)


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
        ax.set_title('Generation vs Food Eaten')
        ax.set_yticks(range(0, max(values) + 1, 1))
        ax.set_xticks(range(1, max(keys) + 1, 1))

        return fig
    
    def save_plot(self, plt):

        # Create the filename
        filename = f'graphs/{self.path}/Generation_vs_Food_plot.png'

        # Save the plot
        plt.savefig(filename)

    #make subdirs for best_snakes and graphs and raw data with selection, crossover method, population size, mutation rate
    def make_subdirs(self):
        selection_method_name = self.selection_method.__name__
        crossover_method_name = self.crossover_method.__name__

        subdir_name = f'{selection_method_name}_{crossover_method_name}_{self.population_size}_{self.mutation_rate}_elitism_{self.elitism_rate}_seed_{self.seed}'

        for parent_dir in ['best_snakes', 'graphs', 'raw_data']:
            path = f'{parent_dir}/{subdir_name}'
            os.makedirs(path, exist_ok=True)

        return subdir_name
    
    def save_score_data(self):
        # Save gen_score_dict to a csv file under path

        filename = f'raw_data/{self.path}/Generation_Food.csv'

        with open(filename, 'w') as f:
            f.write('Generation,Food\n')
            for generation, score in self.gen_best_score_dict.items():
                f.write(f'{generation},{score}\n')

    def save_fitness_data(self):
        filename = f'raw_data/{self.path}/Generation_fitness.csv'

        with open(filename, 'w') as f:
            f.write('Generation,Best Fitness,Avg Fitness\n')
            for generation in self.gen_best_fitness_dict:
                f.write(f'{generation},{self.gen_best_fitness_dict[generation]},{self.gen_avg_fitness_dict[generation]}\n')
