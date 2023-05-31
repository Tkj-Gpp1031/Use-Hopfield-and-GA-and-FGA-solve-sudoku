from tqdm import tqdm

import file_read_GA as f
import numpy as np
import random


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

# Calculating the pheromone between two individuals
'''
The sexual appeal of female fireflies is directly proportional to the number of pheromones released. 
These pheromones reach male fireflies depending on two factors: (i) wind speed and (ii) distance between the selected pair of fireflies. 
Hence, the amount of pheromone for a particular female firefly reaching a corresponding male is formulated [15] as:
Phij=|fi×dis×w|
(3)
Here, dis is calculated by the distance between firefly i and j from the initial parent population, w is any random number within the range of 1 and 9, fi
 is the calculated fitness of firefly i, and Phij is the pheromone calculated for each female firefly towards every male firefly individually 
 and an array is used to store this.
 '''
def distance(male, female):
    i = male
    j = female
    dis = abs(i - j)
    return dis
def calculate_female_pheromone(individual1, distance):
#Calculating female pheromones
    wind_speed = random.randint(1, 9)
    fitness1 = fitness(individual1)
    pheromone = np.abs(fitness1 * distance * wind_speed)
    return pheromone
def calculate_male_brightness(male_fitness, distance, gamma=0.5):
#Calculating male brightness
    brightness = abs(male_fitness * np.exp(-gamma * distance**2))
    return brightness
#Pairing
def calculate_mutual_attraction(population):
    mutual_attractions = []

    for i, female in enumerate(population[:len(population) // 2]):
        for j, male in enumerate(population[len(population) // 2:]):

            dis = distance(i,j)
            muA = calculate_female_pheromone(male['sudoku'],dis) + calculate_male_brightness(fitness(male['sudoku']),dis,gamma=0.5)
            mutual_attractions.append((muA, i, j))

    # Sort the mutual attractions in descending order
    sorted_mutual_attractions = sorted(mutual_attractions, key=lambda x: x[0], reverse=True)



    return sorted_mutual_attractions
#select parents
def select_fireflies(sorted_mutual_attractions, mating_capabilities):
    for firefly_pair in sorted_mutual_attractions:
        male_index = firefly_pair[1]
        female_index = firefly_pair[2]

        if mating_capabilities[male_index] > 0 and mating_capabilities[female_index] > 0:
            mating_capabilities[male_index] -= 1
            mating_capabilities[female_index] -= 1
            return (male_index, female_index)
    return None

#Calculating the maximum number of individual iterations
def mating_capability(fitness, population_size):
    rand_value = random.randint(0, population_size - 1)
    mating_cap = (fitness / 100) * rand_value
    return int(mating_cap)

def selection(population, sorted_mutual_attractions, num_elites):
    selected_pairs = sorted_mutual_attractions[:num_elites]
    elite_indices = [(pair[1], len(population)//2 + pair[2]) for pair in selected_pairs]
    elite_individuals = [population[i] for i in elite_indices]
    return elite_individuals

'''
The crossover process described in the thesis uses a fixed two-point crossover method. The specific steps are as follows:

Starting from the top of the selection table (selection table), firefly pairs are selected for the crossover operation. The pairs are sorted according to mutual attractiveness.
For each selected pair of male and female fireflies, their mating ability is checked. A crossover is only performed if both sexes have a mating ability value greater than zero.
For each pair of fireflies that undergo a crossover, a crossover point is set (fixed after three consecutive sub-grids) and the selected part is then alternately assigned to the new offspring to form a new solution.
After completing the crossover operation, the mating ability value of the biparental fireflies is subtracted by one.
Repeat the above steps until m/2 crossover operations have been performed

Translated with www.DeepL.com/Translator (free version)。'''
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


'''3.3.3.6 Mutation and next generation selection

After the crossover operation, the next stage is neighbourhood-based exchange mutation. In this phase, we perform the following operations for each subgrid:

A random probability is generated. If this probability is less than the pre-determined implicit mutation probability, then we choose two random positions in the current sub-grid to swap.
Before swapping, we need to check whether the two positions are allowed to be swapped. In other words, if the generated position is equal to the index of an existing clue, then this position is not allowed to be swapped.
Each cell belongs to three cells: rows, columns and sub-grids. As there are no duplicate elements in the sub-grid, each cell has 16 neighbouring elements and there may be duplicates in these neighbouring elements.
After generating a pair of positions that allow swapping, we check for duplicates in neighbouring elements. If there is a duplicate in a row or column then we perform a swap operation between these two positions.
This process is performed in all subgrids. It is important to note that in each sub-grid, at most one swap is performed

Translated with www.DeepL.com/Translator (free version)'''

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

# FGA algorithm
def fga(sudoku, max_iter=100, population_size=100, mating_pool_size=50, mutation_probability=0.1, population_reduction_rate=0.99):
    population = initialize_population(sudoku, population_size)

    for iteration in range(max_iter):
        # Check if the best individual has a fitness of 100 (solution found)
        best_individual = max(population, key=lambda x: x['fitness'])
        if best_individual['fitness'] == 100:
            return best_individual['sudoku']
        # Calculate mutual attractions
        mutual_attractions = calculate_mutual_attraction(population)

        # Calculate mating capabilities
        mating_capabilities = [mating_capability(individual['fitness'], population_size) for individual in population]

        # Perform crossover and mutation
        offspring = []
        for _ in range(mating_pool_size // 2):

            parents_indices = select_fireflies(mutual_attractions, mating_capabilities)
            if parents_indices is not None:
                parent1, parent2 = population[parents_indices[0]]['sudoku'], population[parents_indices[1]]['sudoku']
                child_sudoku = crossover(parent1, parent2)
                child_sudoku = mutate_based_on_neighborhood(child_sudoku, sudoku)
                offspring.append({'sudoku': child_sudoku, 'fitness': fitness(child_sudoku)})

        # Combine the parent and offspring population
        population = population + offspring

        # Sort the combined population according to fitness
        population.sort(key=lambda x: x['fitness'], reverse=True)

        # Reduce the population size
        population = population[:int(population_reduction_rate * len(population))]



    return population[0]['sudoku']  # Return the best individual found

def main():
    # Load the sudoku puzzles from the file
    sudoku_puzzles = f.all_sudoku

    # Open the results file
    with open("data/GA_result_dia.txt", "a") as file:

        # For each sudoku puzzle
        for i, puzzle in enumerate(sudoku_puzzles):
            # Optionally, break after the first puzzle for testing purposes
            if i == 100:
                break
            # Solve the puzzle using FGA
            solution = fga(puzzle, max_iter=3000, population_size=100, mating_pool_size=50, mutation_probability = 0.1,population_reduction_rate=0.99)

            if solution is not None:
                print('puzzle:')
                print(puzzle)
                print("Solution:")
                final_solution = solution[0].get('sudoku')
                print(final_solution)
                file.write(f"Sudoku {i}:\n")
                file.write(str(final_solution))
                file.write("\n")
            else:
                print('retry!!!!!!!')
                solution = fga(puzzle, max_iter=3000, population_size=100, mating_pool_size=50, mutation_probability = 0.1,population_reduction_rate=0.99)
                print("Solution:")
                print(solution)
                file.write(f"Sudoku {i}:\n")
                file.write(str(solution))
                file.write("\n")



if __name__ == "__main__":
    main()