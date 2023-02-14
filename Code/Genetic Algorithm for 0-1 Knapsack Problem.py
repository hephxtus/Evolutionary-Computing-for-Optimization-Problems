import random

import numpy
import os
from deap import creator, base, tools, algorithms

IND_INIT_SIZE = 5  # Initial size of the population
# MAX_ITEM = 1000  # Maximum number of items in the knapsack
# MAX_WEIGHT = 1000  # Maximum weight of the knapsack
RUNS = 5

Dataset_dir = "../Datasets/knapsack-data"
Output_dir = "../Output/knapsack-data"

# To assure reproducibility, the RNG seed is set prior to the items
# dict initialization. It is also seeded in main().
convergence = []
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))  # Minimize the weight
creator.create("Individual", set, fitness=creator.Fitness)  # Create the individual as a set of integers


# Create the item dictionary: item name is an integer, and value is
# a (weight, value) 2-tuple.
def loadFromFile(filename: str):
    """Loads the data from the file and returns a list of tuples.
    Each tuple contains the weight and the value of the item.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if line:
            data.append(tuple(map(int, line.split())))
    return data


# Create random items and store them in the items' dictionary.
# for i in range(NBR_ITEMS):
#     items[i] = (random.randint(1, 10), random.uniform(0, 100))


def evalKnapsack(individual):
    """Evaluation function of the knapsack problem. The fitness function
    maximizes the total value of the items in the knapsack.
    """
    weight = 0.0
    value = 0.0
    for item in individual:
        weight += items[item][1]
        value += items[item][0]
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
        return 0, 10000  # Ensure overweighted bags are dominated
    # print(weight, value)
    return value, weight


def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)  # Used in order to keep type
    ind1 &= ind2  # Intersection (inplace)
    ind2 ^= temp  # Symmetric Difference (inplace)
    return ind1, ind2


def mutSet(individual):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(individual) > 0:  # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(len(items)))
    return individual,


import matplotlib.pyplot as plt


def plot_convergence(weight=[], value=[], title="Dataset Name"):
    # plt.plot(weight, label="Weight")
    plt.plot(value, label="Value")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(title)
    plt.savefig(f"{Output_dir}/{title}.png")
    plt.show()


if __name__ == "__main__":
    global MAX_ITEM, MAX_WEIGHT
    NGEN = 100  # Number of generations
    MU = 50  # Number of individuals in population
    LAMBDA = 3 * MU  # Number of children per generation
    CXPB = 0.6  # Probability of crossover
    MUTPB = 0.4  # Probability of mutation
    for file in os.listdir(Dataset_dir):
        items = loadFromFile(Dataset_dir + "/" + file)  # Load the items from the file
        MAX_ITEM = items[0][0]
        MAX_WEIGHT = items[0][1]
        NGEN = MAX_ITEM
        items = items[1:]
        print(MAX_ITEM, MAX_WEIGHT)
        toolbox = base.Toolbox()  # Create a toolbox object
        toolbox.register("attr_item", random.randrange, len(items))  # Create a random item
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_item, MAX_ITEM)  # Create a random individual
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Create a random population

        toolbox.register("evaluate", evalKnapsack)  # Register the evaluation function
        toolbox.register("mate", cxSet)  # Register the crossover function
        toolbox.register("mutate", mutSet)  # Register the mutation function
        toolbox.register("select", tools.selNSGA2)  # Register the selection function

        real_convergence = []
        avg_value = [0] * NGEN
        avg_weight = [0] * NGEN
        for i in range(RUNS):
            random.seed(i)
            pop = toolbox.population(n=MU)  # Create a random population
            hof = tools.ParetoFront()  # Create a hall of fame object
            stats = tools.Statistics(lambda ind: ind.fitness.values)  # Create a statistics object

            stats.register("avg", numpy.mean, axis=0)  # Register the statistics function
            stats.register("std", numpy.std, axis=0)  # Register the statistics function
            stats.register("min", numpy.min, axis=0)  # Register the statistics function
            stats.register("max", numpy.max, axis=0)  # Register the statistics function
            pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                                                     halloffame=hof, verbose=False)  # Run the algorithm
            # log the convergence of the algorithm
            # print(hof[0].fitness.values)
            convergence = logbook.select("max")
            for x in range(NGEN):
                avg_weight[x] += convergence[x][1]
                avg_value[x] += convergence[x][0]
            # real_convergence[-1] += logbook.select("avg")[-1]
            # print(logbook.select("avg"))
            record = stats.compile(pop)
            print(f"{file}- AVG: {record['avg'][0]}, STD: {record['std'][0]}, MAX: {record['max'][0]}")
        avg_value = [x / RUNS for x in avg_value]
        avg_weight = [x / RUNS for x in avg_weight]
        print(avg_value)
        # print(avg_weight)
        # plot_convergence(avg_size)
        plot_convergence(value=avg_value, weight=avg_weight, title=file)

    #     for x in out.select('gen'):
    #         print(x)
    #         exit(1)
    #         # real_convergence[x] += convergence[x]
    #     convergence.clear()
    # for x in range(len(real_convergence)):
    #     real_convergence[x] /= RUNS
    # print(real_convergence)
