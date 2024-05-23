#%%
from pytorch.pytorch_ga_factory import GeneticAlgorithmTorch
from selection_methods import alpha_selection, tournament_selection, elitism_selection, rank_selection, roulette_wheel_selection, top_20_percent
from crossover_methods import single_point_crossover, two_point_crossover, uniform_crossover, arithmetic_crossover
import pygame
import torch


if __name__ == "__main__":
# make combinations of crossovers selection methods and population sizes and mutation rates and run the genetic algorithm

    seeds = range(1)

    population_sizes = [2000]
    mutation_rates = [0.1]
    selection_methods = [alpha_selection, rank_selection, top_20_percent]
    crossover_methods = [single_point_crossover, two_point_crossover, arithmetic_crossover]
    generations = 100
    elistism_rates = [0.1]
    
for seed in seeds:
    torch.manual_seed(seed)
    for population_size in population_sizes:
        for mutation_rate in mutation_rates:
            for selection_method in selection_methods:
                for crossover_method in crossover_methods:
                    for elitism_rate in elistism_rates:
                        ga = GeneticAlgorithmTorch(population_size=population_size, mutation_rate=mutation_rate, number_of_generations=generations, 
                                              selection_method=selection_method, crossover_methods=crossover_method, elitism_rate=elitism_rate, display_best_snake=False)
                        ga.run()


pygame.quit()


#%%
