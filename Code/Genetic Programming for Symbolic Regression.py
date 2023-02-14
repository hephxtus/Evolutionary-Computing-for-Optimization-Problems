# In this question, your task is to build a GP system to automatically evolve a number of genetic
# programs for the following regression problem:
# f (x) = { 1/x + sin x , x > 0
#           2/x + x^2 + 3.0 , x ≤ 0

# You can use a GP library. You should
# • Determine and describe the terminal set and the function set.
# • Design the fitness cases and fitness function.
# • Set the necessary parameters, such as population size, max tree depth, termination
# criteria, crossover and mutation rates.
# • Run the GP system for 3 times with different random seeds. Report the best genetic
# programs (their structure and performance) of each of the 3 runs. Do your observations,
# discussions and draw your conclusions.

# Import libraries
import cmath
import math
import operator
import random

import numpy as np

from deap import gp, creator, base, tools, algorithms


def f(x):
    if x > 0:
        return (1 / x) + np.sin(x)
    else:
        return (2*x) + (x ** 2) + 3.0
# terminal and function set for
# f (x) = { 1/x + sin x , x > 0
#           2/x + x^2 + 3.0 , x ≤ 0

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1, )
# pset = gp.PrimitiveSet()
pset.addPrimitive(operator.add, 2, name="add")
pset.addPrimitive(operator.sub, 2, name="sub")
pset.addPrimitive(operator.mul, 2, name="mul")
pset.addPrimitive(operator.neg, 1, name="neg")
pset.addPrimitive(protectedDiv, 2, name="div")
pset.addPrimitive(np.cos, 1, name="cos")
pset.addPrimitive(np.sin, 1, name="sin")
pset.addPrimitive(lambda x: protectedDiv(1, x), 1, name='inv')
# pset.addPrimitive(, 2, name="pow")
pset.addPrimitive(lambda x: math.sqrt(abs(x)), 1, name="sqrt")

pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x')

# Create the fitness cases
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalFitness(individual, points):
    # Transform the tree expression in a callable function
    # print(individual)
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function
    sqerrors = []
    for x in points:
        try:
            # print(func(x), f(x))
            fn = func(x)
            fi = f(x)
            # convert to real number
            if isinstance(fn, complex):
                fn = fn.real
            if isinstance(fi, complex):
                fi = fi.real
            sqerrors.append((fn - fi) ** 2)
        except Exception as e:
            print(e)
            print(f"Error in individual {individual} for point {x}")
            exit(1)
    return math.fsum(sqerrors) / len(points),

tournament_size = 3
toolbox.register("evaluate", evalFitness, points=np.linspace(-10, 10, 100))
toolbox.register("select", tools.selTournament, tournsize=tournament_size)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))




def SymbolicGP(seed):
    random.seed(seed)


    population_size = 300
    hall_of_fame = 1

    generations = 40

    crossover_rate = 0.7
    mutation_rate = 0.1

    # print(df['x'])


    # Create the population
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(hall_of_fame)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_rate, mutpb=mutation_rate,
                                      ngen=generations, stats=mstats,
                                        halloffame=hof, verbose=False)

    print(f"Best individual for seed {seed} is {hof[0]} with a fitness of {hof[0].fitness.values[0]}")
    return log, hof


def print_header(text, sep='='):
    print(sep * len(text))
    print(text)
    print(sep * len(text))


if __name__ == '__main__':
    for seed in range(1, 4):
        SymbolicGP(seed)
