import random
from game import SnakeGame
from ga_brain import GABrain
import pygame
import matplotlib.pyplot as plt
from selection_methods import top_20_percent, roulette_wheel_selection, rank_selection, tournament_selection
from crossover_methods import single_point_crossover, two_point_crossover, uniform_crossover, arithmetic_crossover

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, selection_method=tournament_selection, crossover_method=two_point_crossover):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
        self.gen_score_dict = {}
        self.selection_method = selection_method
        self.crossover_method = crossover_method

    def initialize_population(self):
        # Create an initial population of random GABrain instances
        return [GABrain() for _ in range(self.population_size)]

    def evaluate_population(self, generation_number: int):
        # Run a game for each GABrain in the population
        highest_amount_of_food_eaten = 0
        best_game: SnakeGame = None
        generation_number = generation_number + 1

        for brain in self.population:
            game = SnakeGame(brain=brain, display=False)
            snake_age, score, food_eaten = game.run()
            brain.set_fitness(snake_age, score)

            if food_eaten > highest_amount_of_food_eaten:
                highest_amount_of_food_eaten = food_eaten
                best_game = game

        print(f"Highest amount of food eaten: {highest_amount_of_food_eaten}")

        # Save the best game's replay to a file
        if best_game is not None:
            best_game.save_game_states(f'best_snakes/Gen_{generation_number}_snake.pkl')
        
        # Replay the best game
        if best_game is not None:
            best_game.play_back(f'best_snakes/Gen_{generation_number}_snake.pkl')

        #save to dictionary
        self.gen_score_dict[generation_number] = highest_amount_of_food_eaten


    def mutation(self, child):
        # Randomly change some genes in the child's genome
        for i in range(len(child.genome)):
            if random.random() < self.mutation_rate:
                child.genome[i] = child.random_gene()

    def crossover(self, parent1, parent2):
        return self.crossover_method(parent1, parent2)
    
    def selection(self):
        return self.selection_method(self.population)
    
    def generate_new_population(self):
        # Generate new brains until we reach the original population size
        new_population = []

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
            self.evaluate_population(generation)

            best_brain = max(self.population, key=lambda brain: brain.fitness)
            # print(f"Generation {generation}: Best fitness = {best_brain.fitness}")

            ga.generate_new_population()
    
    def make_plot(self):
        keys = list(self.gen_score_dict.keys())
        values = [int(value) for value in self.gen_score_dict.values()]
        plt.scatter(keys, values)
        plt.plot(keys, values)
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.title('Generation vs Score')
        plt.yticks(range(0, max(values)+1, 1))  # Set y-ticks
        plt.xticks(range(1, max(keys)+1, 1))  # Set x-ticks starting from 1
        plt.show()

if __name__ == "__main__":
    NUM_GENERATIONS = 30
    POPULATION_SIZE = 2000
    MUTATION_RATE = 0.2

    ga = GeneticAlgorithm(POPULATION_SIZE, MUTATION_RATE, selection_method=tournament_selection, crossover_method=two_point_crossover)
    ga.run()
    ga.make_plot()

    pygame.quit()  # Add this line