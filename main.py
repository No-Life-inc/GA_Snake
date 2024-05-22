from ga_factory import GeneticAlgorithm
from selection_methods import alpha_selection, tournament_selection, elitism_selection, rank_selection, roulette_wheel_selection, top_20_percent
from crossover_methods import single_point_crossover, two_point_crossover, uniform_crossover, arithmetic_crossover



if __name__ == "__main__":
# make combinations of crossovers selection methods and population sizes and mutation rates and run the genetic algorithm

    population_sizes = [1000, 2000, 3000]
    mutation_rates = [0.05, 0.1, 0.15]
    selection_methods = [alpha_selection, tournament_selection, elitism_selection, rank_selection, roulette_wheel_selection, top_20_percent]
    crossover_methods = [single_point_crossover, two_point_crossover, uniform_crossover, arithmetic_crossover]
    generations = 100

    for population_size in population_sizes:
        for mutation_rate in mutation_rates:
            for selection_method in selection_methods:
                for crossover_method in crossover_methods:
                    ga = GeneticAlgorithm(population_size, mutation_rate, generations,selection_method, crossover_method)
                    ga.run()