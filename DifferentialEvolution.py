"""
Implementation of the Differential Evolution algorithm
"""

import numpy as np

class DiffEvo:
    def __init__(self,bounds, mut=0.8, crossp=0.7, popsize=20, its=1000, problem_dim=2):
        population = np.random.rand(popsize, problem_dim)
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + population * diff
        fitness = np.asarray([fobj(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]



        for i in range(its):
            for j in range(popsize):
                idxs = [idx for idx in range(popsize) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + mut * (b - c), 0, 1)
                cross_points = np.random.rand(problem_dim) < crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, problem_dim)] = True
                trial = np.where(cross_points, mutant, population[j])
                trial_denorm = min_b + trial * diff
                f = fobj(trial_denorm)
                if f < fitness[j]:
                    fitness[j] = f
                    population[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm
            yield best, fitness[best_idx]