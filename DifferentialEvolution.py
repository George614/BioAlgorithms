"""
Implementation of the Differential Evolution algorithm
"""

import numpy as np

def diffevo(obj_func, bounds, F=0.8, Cr=0.7, pop_size=20, iterations=10):
    dims = len(bounds)
    population = np.random.rand(pop_size, dims)
    lower_bound, upper_bound = np.asarray(bounds).T
    range_ = np.fabs(lower_bound - upper_bound)
    pop_denorm = lower_bound + population * range_
    fitness = np.asarray([obj_func(ind) for ind in pop_denorm])
    idx_best = np.argmin(fitness)
    solution_best = pop_denorm[idx_best]
    for i in range(iterations):
        for j in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != j]
            # pick three vectors from the population
            p, q, r = population[np.random.choice(idxs, 3, replace=False)]
            # generate donate vector and clamp it between 0 and 1
            v = np.clip(p + F * (q - r), 0, 1)
            # carry out binomial crossover scheme
            ran_vec = np.random.rand(dims)
            cross_points = ran_vec < Cr  # Cr is the crossover parameter
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dims)] = True
            # create new solution
            u = np.where(cross_points, v, population[j])
            u_denorm = lower_bound + u * range_
            # evaluate solution
            f = obj_func(u_denorm)
            # select and update solution
            if f < fitness[j]:
                fitness[j] = f
                population[j] = u
                if f < fitness[idx_best]:
                    idx_best = j
                    solution_best = u_denorm
        yield solution_best, fitness[idx_best]
