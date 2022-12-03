"""
Assignment Title: Implementation of Standard Genetic Algorithm to solve the Traveling Salesman Problem
Purpose: To solve the Traveling Salesman Problem with GA
Language: Implementation in Python

Author: Hemant Ramphul
Github: https://github.com/hemantramphul/
Date: 19 November 2022

Université des Mascareignes (UdM)
Faculty of Information and Communication Technology
Master Artificial Intelligence and Robotics
Official Website: https://udm.ac.mu
"""

import sys
from City import City  # import create class City as City
import matplotlib.pyplot as plt
import random as rand


def tour_length(tour):
    """
    :type tour: list of City objects
    :rtype calculates the length (or cost) of the tour
    """
    length_so_far = 0.0
    for i in range(0, len(tour) - 2):
        length_so_far += tour[i].distance(tour[i + 1])
    return length_so_far


def generate_rand_tour(list_of_cities):
    """
    Used to generate a rand tour for the initial population creation.

    :param list_of_cities: list of City objects
    :rtype a rand tour of cities where tour[0] == tour[len(tour)-1].

    return type:list of City objects
    """
    tour = list_of_cities[1:]  # take city 1 to the end
    rand.shuffle(tour)  # reorganize the order of the city
    tour.insert(0, list_of_cities[0])  # insert value at the specified position
    tour.append(list_of_cities[0])  # place new city in the available space
    return tour  # return tour


def generate_population(list_of_cities, population_size):
    """
    Generates the initial population of the genetic algorithm.
    Loop over the new population's size and create individuals from current population.

    :param population_size: (type: Int): the size of the population.
    :param list_of_cities: list of City objects
    :rtype list of a list of City objects.
        a population is a list of tours, which are lists of City objects
    """

    population = []  # create an empty array
    for h in range(0, population_size):  # iterate using a for loop with population_size
        rand_tour = generate_rand_tour(list_of_cities)  # call method generate_rand_tour for each population_size
        population.append(rand_tour)  # append the city to the end of the list
    return population  # return population


"""
SELECTION METHODS
"""


def selection(population):
    """
    The selection method over population

    :param population:
    :rtype: object
    """
    population_size = len(population)  # find population size by using len()
    survivors = []  # the generation undergoes "natural selection"
    tour_costs = []  # saves a tour

    for r in range(0, population_size):  # iterate using a for loop with population_size
        tour_costs.append(1.0 / tour_length(population[r]))  # saves a tour

    current_value = 0
    probability_ranges = []
    for tour in range(0, len(tour_costs)):  # iterate using a for loop with tour_costs
        range_tuple = (current_value, current_value + tour_costs[tour])
        probability_ranges.append(range_tuple)
        current_value += tour_costs[tour]

    # selection part
    # select individuals from population
    assert sum(tour_costs) == current_value  # program will raise an AssertionError, if not

    for r in range(0, population_size):  # iterate using a for loop with population_size
        selection_number = rand.random() * current_value  # number for semi-rand selection
        index = 0
        for prob_range in probability_ranges:  # iterate using a for loop with probability_ranges
            if prob_range[0] <= selection_number <= prob_range[1]:
                survivors.append(population[index])  # add to survivor list
            index += 1
    assert len(survivors) == population_size  # program will raise an AssertionError, if not
    survivors.sort(key=tour_length)  # sort survivors

    return survivors  # return survivors


def classical_selection(population):
    """
    The selection method over population

    :param population:
    :rtype: survivors
    """
    survivors = []  # survivors list
    population_size = len(population)  # length of population

    for po in range(0, population_size):  # iterate using a for loop with population_size
        selection_range = 10
        # random city from range 10
        rand_index = rand.randint(1, population_size - 2 - selection_range)
        # get best tour
        best_tour = population[rand_index]
        # loop to find the tour
        for the_index in range(rand_index + 1, rand_index + selection_range + 1):
            if tour_length(population[the_index]) < tour_length(best_tour):
                best_tour = population[the_index]
        survivors.append(best_tour)  # append the survivor
    return survivors


def generate_children(survivors, mutation_rate):
    """
    Generate a new children generation from survivors

    :param survivors: List of city objects).
    :rtype a population.  Type: list of tours, which are lists of City objects.
    """
    population_size = len(survivors)  # length of survivors
    children = []  # create a children list
    # Loop and create children
    for index in range(0, population_size, 2):
        # index is parent1's index, index+1 is parent2's index
        parent1 = survivors[index]  # get the city at target position
        parent2 = survivors[index + 1]  # get the city at target position

        # Crossover parents
        child1 = crossover(parent1, parent2)  # swap them around
        child2 = crossover(parent2, parent1)  # swap them around

        if rand.random() < mutation_rate:  # apply mutation rate
            child1 = mutate(child1)

        if rand.random() < mutation_rate:  # apply mutation rate
            child2 = mutate(child2)  # second random position in the tour

        # add to array list
        children.append(child1)
        children.append(child2)

    return children


def evolve(population, mutation_rate):
    """
    Evolves a population over one generation.

    :param population: The population (list of tours, which are lists of City objects).
    :rtype a population.  Type: list of tours, which are lists of City objects.
    """
    population_size = len(population)  # length of survivors
    survivors = selection(population=population)  # call method selection with population as argument

    next_generation = generate_children(survivors, mutation_rate=mutation_rate)

    mix = []  # mix both survivor and next generation
    mix.extend(survivors)  # add survivor to list
    mix.extend(next_generation)  # add generation to list

    mix.sort(key=tour_length)  # make a sort

    return mix[:population_size]


"""
CROSSOVER METHODS
"""


def cycle_crossover(parent1, parent2):
    # find city
    def index_of(the_city, tour):
        for index in range(1, len(tour) - 1):
            if tour[index] is the_city:
                return index
        return -1

    len_of_tour = len(parent1)  # parent1 length
    # get the cities at target position in tour
    child = [None] * len_of_tour
    rand_index = rand.randint(1, len_of_tour - 2)

    value_at_initial_index_of_parent1 = parent1[rand_index]
    value_at_index_of_parent1 = parent1[rand_index]
    child[rand_index] = value_at_index_of_parent1
    value_at_index_of_parent2 = parent2[rand_index]
    set_of_indices = {rand_index}
    while value_at_index_of_parent2 is not value_at_initial_index_of_parent1:  # I changed from != to is not
        # print("iteration")
        index_in_parent1 = index_of(value_at_index_of_parent2, parent1)
        child[index_in_parent1] = parent1[index_in_parent1]
        value_at_index_of_parent2 = parent2[index_in_parent1]
        set_of_indices.add(index_in_parent1)

    for i in range(1, len_of_tour - 1):
        if i not in set_of_indices:
            child[i] = parent2[i]

    child[0] = parent1[0]
    child[len_of_tour - 1] = parent1[len_of_tour - 1]
    return child


def crossover(tour1, tour2):
    """
    Crossover to a set of parents and creates offspring

    :param tour1:
    :param tour2:
    :rtype
    """
    len_of_tours = len(tour1)  # tour1 length
    # get the cities at target position in tour1
    start_pos = rand.randint(1, len_of_tours - 2)  # get position in the tour1
    end_pos = rand.randint(1, len_of_tours - 2)  # get position in the tour1

    while start_pos == end_pos:
        end_pos = rand.randint(1, len_of_tours - 1)
    temp = start_pos
    if start_pos > end_pos:
        start_pos = end_pos
        end_pos = temp

    # now I have gotten the range that will be the crossover section
    # this is the section that will be taken from tour 1 and given to the child
    child_tour = [None] * len_of_tours
    child_tour[0] = tour1[0]
    child_tour[-1] = tour1[-1]
    for i in range(start_pos, end_pos + 1):
        child_tour[i] = tour1[i]

    tour2_index = 1
    # child_index = 1
    for i in range(tour2_index, len_of_tours - 1):  # loop through tour2
        child_contains_city_of_tour2_at_i = False
        for k in range(1, len_of_tours - 1):
            if child_tour[k] is None:
                continue
            if child_tour[k] is tour2[i]:
                child_contains_city_of_tour2_at_i = True
                break
        if not child_contains_city_of_tour2_at_i:
            for j in range(1, len_of_tours - 1):
                if child_tour[j] is None:
                    child_tour[j] = tour2[i]
                    break
    return child_tour


"""
MUTATE METHOD
"""


def swap(tour, first_element, second_element):
    """
    Takes the tour input and return a new tour identical to the input tour except the items
    from first_element to second_element are in reverse order

    :param tour: list of City Objects
    :param i: first index
    :param k: second index
    :rtype a new tour, which is a list of City objects
    """
    new_tour = []  # new tour
    new_tour.extend(tour[:first_element])  # adds the first elements to the end of the city list
    cutlet = tour[first_element:second_element]  # slice element in the array
    new_tour.extend(cutlet[::-1])  # list's last element
    new_tour.extend(tour[second_element:])  # adds the second elements to the end of the city list

    return new_tour


def mutate(tour):
    """
    Mutate a tour using swap mutation

    :param tour: list of City Objects
    """
    number_of_cities_in_a_tour = len(tour)  # tour length
    # get the cities at target position in tour
    first_element = rand.randint(1, number_of_cities_in_a_tour - 2)  # first random position in the tour
    second_element = rand.randint(1, number_of_cities_in_a_tour - 2)  # second random position in the tour
    # if first element equal the number_of_cities_in_a_tour then the first element will decrease by one
    if first_element == number_of_cities_in_a_tour:
        first_element -= 1
    # if first element equal the second element then the second element will increase by one
    if first_element == second_element:
        second_element += 1
    # else if second element is less than first element,
    # the first element will equal the second element and second element will equal the first element
    elif second_element < first_element:
        # swap them around
        temp = first_element
        first_element = second_element
        second_element = temp

    # let swap each other to get a new tour
    new_tour = swap(tour, first_element, second_element)

    return new_tour


"""
MAIN PROGRAM
"""

POPULATION_SIZE = 200
MUTATION_RATE = 0.01
CITIES = 10
ITERATIONS = 0
LIMIT = 1000

# Create city list in term array
cityList = []
# For graphing
cost = []

# Create city randomly using City class
for i in range(0, CITIES):
    cityList.append(City(x=int(rand.random() * 200), y=int(rand.random() * 200), num=i))

cities_list = cityList

population = generate_population(cities_list, population_size=POPULATION_SIZE)
the_best = population[0]
len_pop = len(population)

for i in range(1, len_pop - 1):
    if tour_length(population[i]) < tour_length(the_best):
        the_best = population[i]

while ITERATIONS < LIMIT:
    population = evolve(population=population, mutation_rate=MUTATION_RATE)

    if tour_length(population[0]) < tour_length(the_best):
        the_best = population[0]
    ITERATIONS += 1
    cost.append(tour_length(population[0]))
    print(f"\rSolving TSP: {(ITERATIONS / LIMIT) * 100:.1f} %", sep='', end='', file=sys.stdout, flush=False)

# Break line
print("\n")

for city in the_best:
    print(f"{city}")

print("\nBest tour cost: ", tour_length(the_best))
print("Iterations: ", ITERATIONS)
print("Population size: ", POPULATION_SIZE)
print("Mutation rate: ", MUTATION_RATE)

figure = plt.figure()
plt.plot(cost)
figure.suptitle(f"GA: Cost of Best Individual for {CITIES} cities", fontsize=12)
plt.xlabel("Iterations", fontsize=10)
plt.ylabel("Cost", fontsize=10)
axes = plt.gca()
axes.set_xlim([0, len(cost)])
axes.set_ylim([0, max(cost)])
plt.show()  # show the graph
