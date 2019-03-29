"""
Implementation of the Differential Evolution algorithm
"""

import numpy as np

class DiffEvo:
    def __init__(self, obj_func, bounds, F=0.8, Cr=0.7, popsize=20, iterations=1000):
        self.dimensions = len(bounds)
        self.pop = np.random.rand(popsize, self.dimensions)
        min_b, max_b = np.asarray(bounds).T
        range_ = np.fabs(min_b - max_b)
        pop_denorm = min_b + self.pop * range_
        fitness = np.asarray([obj_func(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        for i in range(iterations):
            for j in range(popsize):
                idxs = [idx for idx in range(popsize) if idx != j]
                p, q, r = self.pop[np.random.choice(idxs, 3, replace=False)]
                # generate our donate vector and clamp it between 0 and 1
                v = np.clip(p + F * (q - r), 0, 1)
                # carry out binomial crossover scheme
                cross_points = np.random.rand(self.dimensions) < Cr  # crossover parameter
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimensions)] = True
                trial = np.where(cross_points, v, self.pop[j])
                trial_denorm = min_b + trial * range_
                f = obj_func(trial_denorm)
                if f < fitness[j]:
                    fitness[j] = f
                    self.pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm
            yield best, fitness[best_idx]


if __name__ == '__main__':
    result = list(de(lambda x: sum(x**2)/len(x), bounds=[(-100, 100)] * 8))
    print(result[-1])
