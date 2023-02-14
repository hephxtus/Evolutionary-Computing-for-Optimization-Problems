"""
In this question, your task is to use particle swarm optimisation (PSO) to search for the
minimum of the following two functions:
    1. Rosenbrock’s function, f(x, y) = (1 − x)2 + 100(y − x2)2
    2. Griewanks’s function, f(x, y) = 1 + (x2 + y2)/4000 − cos(x)cos(y/√2)
where D is the number of variables, i.e. x1, x2, ..., xD.

For D = 20, do the following:
• Choose appropriate values for c1, c2, w, and population size.
• Determine the fitness function, particle encoding, topology type, and stopping criterion
in PSO.
• Since PSO is a stochastic process, for each function, f1(x) or f2(x), repeat the experiments 30 times, and report the mean and standard deviation of your results, i.e. f1(x)
or f2(x) values.
• Analyse your results, and draw your conclusions.
Then for D = 50, solve the Griewanks’s function using the same settings in PSO (repeat 30
times). Report the mean and standard deviation of your results. Compare the performance
of PSO on the Griewanks’s function when D = 20 and D = 50. Analyse your results, and
draw you conclusion
"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pymoo.algorithms.soo.nonconvex.pso import PSO as PSO2
from pymoo.problems.single import Rastrigin
from pymoo.optimize import minimize


def f1(*x):
    """
    x: list of float
    :param x:
    :param y:
    :return:
    """
    return sum(((1 - x[i]) ** 2 + 100 * ((x[i + 1] - x[i] ** 2) ** 2)) for i in range(len(x) - 1))
    # return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


# Griewanks's function
def f2(*x):
    """
    x: list of float
    :param x:
    :param y:
    :return:
    """
    # ii = range(1, len(x))
    # s = sum([xi ** 2 for xi in x] / 4000)
    # prod = np.prod(np.cos(x / math.sqrt(ii)))
    #
    # y = sum - prod + 1
    # return (y)
    s = sum([xi ** 2 for xi in x]) / 4000
    prod = np.prod([np.cos(x[i] / math.sqrt(i + 1)) for i in range(len(x))])
    return s - prod + 1
    # return sum((xi ** 2 + xii ** 2) / 4000 - math.cos(xi) * math.cos(xii / math.sqrt(2)) + 1 for xi, xii in x)
    # return 1 + (x ** 2 + y ** 2) / 4000 - math.cos(x) * math.cos(y / math.sqrt(2))


# Particle Swarm Optimisation
def PSO(func, D, pa, ga, w, N, max_iter, bounds, target_fitness, error_threshold=1000):
    """
    :param func: (function) function to be optimised
    :param D: (int) Number of variables
    :param pa: (float) acceleration coefficient 1
    :param ga: (float) acceleration coefficient 2
    :param w: (float) inertia coefficient
    :param N: (int) population size
    :param max_iter: (int) maximum number of iterations
    :return: ga_best_fitness, ga_best_position, convergence_curve
    """
    # Initialise the population
    population = []
    step = (w-0.4)/max_iter
    for i in range(N):
        position = []
        for j in range(D):
            position.append(random.uniform(bounds[0], bounds[1]))
        velocity = [0] * D
        fitness = func(*position)
        p_best_position = position
        p_best_fitness = fitness
        particle = {'position': position, 'velocity': velocity, 'fitness': fitness, 'p_best_position': p_best_position,
                    'p_best_fitness': p_best_fitness}
        population.append(particle)

    # Sort the population by fitness
    population = sorted(population, key=lambda x: x['fitness'])

    # Initialise the global best
    gbest_position = population[0]['position']
    gbest_fitness = population[0]['fitness']

    # Initialise the convergence curve
    convergence = []
    convergence.append(gbest_fitness)

    # Iterate until the maximum number of iterations is reached
    for i in range(max_iter):
        for j in range(N):
            # Update the velocity
            new_velocity = []
            for k in range(D):
                r1 = random.random()
                r2 = random.random()

                current_velocity = w * population[j]['velocity'][k]
                cognitive = pa * r1 * (population[j]['p_best_position'][k] - population[j]['position'][k])
                social = ga * r2 * (gbest_position[k] - population[j]['position'][k])
                new_velocity.append(current_velocity + cognitive + social)
            population[j]['velocity'] = new_velocity

            # Update the position
            new_position = []
            for k in range(D):
                new_position.append(population[j]['position'][k] + population[j]['velocity'][k])
            population[j]['position'] = new_position

            # Update the fitness
            fitness = func(*new_position)
            population[j]['fitness'] = fitness

            # Update the p_best and g_best
            if abs(fitness-target_fitness) < abs(population[j]['p_best_fitness'] - target_fitness):
                population[j]['p_best_position'] = new_position
                population[j]['p_best_fitness'] = fitness

                if abs(fitness-target_fitness) < abs(gbest_fitness-target_fitness):
                    gbest_position = new_position
                    gbest_fitness = fitness

        # Append the g_best fitness to convergence curve
        convergence.append(abs(gbest_fitness-target_fitness))
        error = abs(target_fitness - gbest_fitness)
        if error <= error_threshold:
            break
        w -= step

    return gbest_fitness, gbest_position, convergence


# Plot the convergence curve
def plot_convergence_curve(convergence_curve, title):
    plt.plot(convergence_curve)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (abs error)')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    min_rosenbrock = []
    min_griewanks = []

    # set the parameters
    D = 20
    c1 = 1.49618
    c2 = 0.49618
    w = 0.9
    N = 500
    max_iter = 50
    bounds = [-30, 30]
    Rosenbrock_target_fitness = f1(*[1] * D)
    Griewanks_target_fitness = f2(*[0]*D)

    print("Rosenbrock target fitness: ", Rosenbrock_target_fitness)
    print("Griewanks target fitness: ", Griewanks_target_fitness)
  # Rosenbrock's function
    convergence_curve = [[]] * max_iter
    for i in range(30):
        gbest_fitness, gbest_position, convergence = PSO(f1, D, c1, c2, w, N, max_iter, bounds,
                                                         target_fitness=Rosenbrock_target_fitness,
                                                         error_threshold=10 * D)
        min_rosenbrock.append(gbest_fitness)
        convergence_curve = [convergence_curve[c] + [convergence[c]] if c < len(convergence) else [] for c in
                             range(max_iter)]
    convergence_curve = list(filter(None, convergence_curve))
    convergence_curve = list(np.mean(c) for c in convergence_curve)

    plot_convergence_curve(convergence_curve, title='Rosenbrock\'s function (D={})'.format(D))

    # Griewanks's function
    convergence_curve = [[]] * max_iter
    for i in range(30):
        gbest_fitness, gbest_position, convergence = PSO(f2, D, c1, c2, w, N, max_iter, bounds,
                                                         target_fitness=Griewanks_target_fitness,
                                                         error_threshold=Griewanks_target_fitness * 0.01 * D)
        min_griewanks.append(gbest_fitness)
        convergence_curve = [convergence_curve[c] + [convergence[c]] if c < len(convergence) else [] for c in
                             range(max_iter)]
    convergence_curve = list(filter(None, convergence_curve))
    convergence_curve = list(np.mean(c) for c in convergence_curve)
    plot_convergence_curve(convergence_curve, title='Griewanks\'s function (D={})'.format(D))

    print("")

    # print mean and std of function
    print("Rosenbrock's function for D={}:".format(D))
    print("mean: ", np.mean(min_rosenbrock))
    print("std: ", np.std(min_rosenbrock))

    print("")

    # print mean and std of function
    print("Griewanks's function for D={}:".format(D))
    print("mean: ", np.mean(min_griewanks))
    print("std: ", np.std(min_griewanks))

    # Then for D = 50, solve the Griewanks’s function using the same settings in PSO (repeat 30
    # times). Report the mean and standard deviation of your results.
    print("")
    min_rosenbrock = []
    min_griewanks = []

    # set the parameters
    D = 50

    # Rosenbrock's function
    # convergence_curve = [[]] * max_iter

    # for i in range(30):
    #     gbest_fitness, gbest_position, convergence = PSO(f1, D, pa, ga, w, N, max_iter, bounds,
    #                                                      target_fitness=Rosenbrock_target_fitness,
    #                                                      error_threshold=10 * D)
    #     min_rosenbrock.append(gbest_fitness)
    #     convergence_curve = [convergence_curve[c] + [convergence[c]] if c < len(convergence) else [] for c in
    #                          range(max_iter)]
    # convergence_curve = list(filter(None, convergence_curve))
    # convergence_curve = list(np.mean(c) for c in convergence_curve)
    # plot_convergence_curve(convergence_curve, title='Rosenbrock\'s function (D={})'.format(D))

    # Griewanks's function

    convergence_curve = [[]] * max_iter
    for i in range(30):
        gbest_fitness, gbest_position, convergence = PSO(f2, D, c1, c2, w, N, max_iter, bounds,
                                                         target_fitness=Griewanks_target_fitness,
                                                         error_threshold=Griewanks_target_fitness * 0.01 * D)
        min_griewanks.append(gbest_fitness)
        convergence_curve = [convergence_curve[c] + [convergence[c]] if c < len(convergence) else [] for c in
                             range(max_iter)]
    convergence_curve = list(filter(None, convergence_curve))
    convergence_curve = list(np.mean(c) for c in convergence_curve)
    plot_convergence_curve(convergence_curve, title='Griewanks\'s function (D={} )'.format(D))

    # print mean and std of function
    # print("Rosenbrock's function for D={}:".format(D))
    # print("mean: ", np.mean(min_rosenbrock))
    # print("std: ", np.std(min_rosenbrock))

    # print mean and std of function
    print("Griewanks's function for D={}:".format(D))
    print("mean: ", np.mean(min_griewanks))
    print("std: ", np.std(min_griewanks))
