import random


def top_20_percent(population):
    sorted_population = sorted(population, key=lambda brain: brain.fitness, reverse=True)
    top_20_percent = sorted_population[:int(0.2 * len(sorted_population))]
    return random.choice(top_20_percent)


def roulette_wheel_selection(population):
    total_fitness = sum(brain.fitness for brain in population)
    selected_value = random.uniform(0, total_fitness)
    cumulative_fitness = 0
    for brain in population:
        cumulative_fitness += brain.fitness
        if cumulative_fitness >= selected_value:
            return brain


def rank_selection(population):
    sorted_population = sorted(population, key=lambda brain: brain.fitness, reverse=True)
    rank_sum = len(population) * (len(population) + 1) / 2
    selected_rank = random.randint(1, len(population))
    cumulative_rank = 0
    for i, brain in enumerate(sorted_population):
        cumulative_rank += i + 1
        if cumulative_rank >= selected_rank:
            return brain


def tournament_selection(population):
    tournament_size = min(len(population), 3)
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda brain: brain.fitness)


def elitism_selection(population, elitism_rate=0.1):
    sorted_population = sorted(population, key=lambda brain: brain.fitness, reverse=True)
    num_elites = int(len(population) * elitism_rate)
    return sorted_population[:num_elites]


def alpha_selection(population):
    # Sort the population by fitness
    sorted_population = sorted(population, key=lambda brain: brain.fitness, reverse=True)
    # Select the best individual as alpha
    alpha = sorted_population[0]
    # Select another individual from the top 20%
    top_20_percent = sorted_population[:int(0.2 * len(sorted_population))]
    mother = random.choice(top_20_percent)
    return alpha, mother