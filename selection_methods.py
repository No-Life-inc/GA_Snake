import random


def truncation_selection(population):
    top_20_percent = population[:int(0.2 * len(population))]
    return top_20_percent


def roulette_wheel_selection(population):
    total_fitness = sum(brain.fitness for brain in population)
    selected_value = random.uniform(0, total_fitness)
    cumulative_fitness = 0
    for brain in population:
        cumulative_fitness += brain.fitness
        if cumulative_fitness >= selected_value:
            return brain


def rank_selection(population, num_selected):
    selected_individuals = []
    for _ in range(num_selected):
        selected_rank = random.randint(1, len(population))
        cumulative_rank = 0
        for i, brain in enumerate(population):
            cumulative_rank += i + 1
            if cumulative_rank >= selected_rank:
                selected_individuals.append(brain)
                population.remove(brain)  # Remove the selected individual from the population
                break
    return selected_individuals


def tournament_selection(population):
    tournament_size = min(len(population), 3)
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda brain: brain.fitness)


def elitism_selection(population, elitism_rate=0):
    num_elites = int(len(population) * elitism_rate)
    return population[:num_elites]


def alpha_selection(population):
    alpha = population[0]
    top_20_percent = population[:int(0.2 * len(population))]
    return alpha, top_20_percent