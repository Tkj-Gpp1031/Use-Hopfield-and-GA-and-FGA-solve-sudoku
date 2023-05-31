import time

from tqdm import tqdm

import file_read_GA as f
import numpy as np
import random

#init
def initialize_population(sudoku, population_size):
    population = []
    for _ in range(population_size):
        individual = np.copy(sudoku)
        for subgrid_x in range(3):
            for subgrid_y in range(3):
                subgrid_values = set(individual[3*subgrid_x:3*(subgrid_x+1), 3*subgrid_y:3*(subgrid_y+1)].flatten())
                to_fill = np.where(individual[3*subgrid_x:3*(subgrid_x+1), 3*subgrid_y:3*(subgrid_y+1)] == 0)
                to_fill_rows = to_fill[0] + 3*subgrid_x
                to_fill_cols = to_fill[1] + 3*subgrid_y
                to_fill_values = list(set(range(1, 10)) - subgrid_values)
                individual[to_fill_rows, to_fill_cols] = np.random.choice(to_fill_values, size=len(to_fill_rows), replace=False)
                # Add Sudoku clues, i.e. fill in the identified numbers directly

        individual_fitness = fitness(individual)
        individual_info = {'sudoku': individual, 'fitness': individual_fitness}
        population.append(individual_info)
    return population


def fitness(individual):
    score = 0
    individual = np.array(individual)

    for i in range(9):
        # Calculating row and column repetitions
        row_set = set(individual[i, :])
        row_lst = individual[i, :].tolist()
        col_set = set(individual[:, i])
        col_lst = individual[:, i].tolist()

        for item in row_set:
            if row_lst.count(item) > 1:
                score += 1

        for item in col_set:
            if col_lst.count(item) > 1:
                score += 1

    # Calculating the repetition of subgrids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            block_set = set(individual[i:i + 3, j:j + 3].flatten())
            block_lst = individual[i:i + 3, j:j + 3].flatten().tolist()
            for item in block_set:
                if block_lst.count(item) > 1:
                    score += 1

    # Boxes
    score = 100 - score
    return score

# 计算两个个体之间的信息素

def mating_capability(fitness, population_size):
    rand_value = random.randint(0, population_size - 1)
    mating_cap = (fitness / 100) * rand_value
    return mating_cap

def selection(population, fitness_scores, mating_capabilities, num_elites):
    indices = np.argsort(fitness_scores)
    elite_indices = indices[-num_elites:]
    elite_indices = sorted(elite_indices, key=lambda idx: mating_capabilities[idx], reverse=True)
    return [population[i] for i in elite_indices]

'''
交叉过程，采用了固定两点交叉方法。具体步骤如下：。'''
def crossover(parent1, parent2):
    child = np.copy(parent1)

    # Set up crossover points
    crossover_points = [3, 6]

    # Two-point crossover for each parent
    for i, cp in enumerate(crossover_points[:-1]):
        next_cp = crossover_points[i + 1]
        if i % 2 == 0:
            child[cp:next_cp, :] = parent2[cp:next_cp, :]
        else:
            child[:, cp:next_cp] = parent2[:, cp:next_cp]


    return child


'''3.3.3.6 Mutations and next generation selection。'''

def mutate_based_on_neighborhood(individual, initial_sudoku):
    initial_sudoku = np.array(initial_sudoku)
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            if random.random() < 0.1:
                allowed_positions = [(x, y) for x in range(i, i + 3) for y in range(j, j + 3) if initial_sudoku[x, y] == 0]
                if len(allowed_positions) < 2:
                    continue
                pos1, pos2 = random.sample(allowed_positions, 2)

                if any(individual[pos1[0], :] == individual[pos2[0], pos2[1]]) or any(
                        individual[:, pos1[1]] == individual[pos1[0], pos2[1]]):
                    individual[pos1], individual[pos2] = individual[pos2], individual[pos1]

    return individual

# FGA算法
def ga(sudoku, max_iter=3000, population_size=100, num_elites=2, mating_pool_size=50,population_reduction_rate=0.99):
    population = initialize_population(sudoku,population_size)

    for iteration in range(max_iter):
        current_population_size = int(population_size * (population_reduction_rate ** iteration))

        fitness_scores = [fitness(individual['sudoku']) for individual in population]
        best_individual = min(population, key=lambda x: x['fitness'])
        if best_individual['fitness'] == 100:
            return best_individual['sudoku']

        elites = selection(population, fitness_scores, [mating_capability(f, population_size) for f in fitness_scores], num_elites)

        new_population = [elites[0]]

        # Select total number of pairs
        for _ in range(mating_pool_size):
            parents = selection(population, fitness_scores, [mating_capability(f, population_size) for f in fitness_scores], num_elites - 1)
            child = crossover(parents[0]['sudoku'], parents[1]['sudoku'])
            mutated_child = mutate_based_on_neighborhood(child,sudoku)
            if fitness(mutated_child) > fitness(child):
                child = mutated_child
            new_population.append({'sudoku': child, 'fitness': fitness(child)})

        population = new_population

    return population

# 测试部分

def main():
    # Load the sudoku puzzles from the file
    sudoku_puzzles = f.all_sudoku

    # Open the results file
    f_ = open("data/normal_GA_time.txt", "a")
    f_.write('ez_time:')
    # For each sudoku puzzle
    for i, puzzle in enumerate(sudoku_puzzles):
            # Optionally, break after the first puzzle for testing purposes
        if i == 100:
            break
            # Solve the puzzle using FGA
        ticks = time.time()
        solution = ga(puzzle, max_iter=5000, population_size=200, num_elites=4,population_reduction_rate=0.99)

        if solution is not None:
            ticks_end = time.time()
            final_solution = solution[0].get('sudoku')
            runing_time = ticks_end - ticks
            f_.write(str(runing_time)+'\n')







if __name__ == "__main__":
    main()