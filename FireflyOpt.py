"""
Find optima of a given function using firefly algorithm
"""

from Firefly import Firefly
import operator

class FireflyOptimizer:
    def __init__(self, obj_func, bounds, pop_size=10, dims=2, max_iters=50, alpha=0.25, beta_0=1, gamma=0.97):
        self.benchmark = obj_func
        self.max_iter = max_iters
        self.best = None
        self.population = [Firefly(alpha, beta_0, gamma, bounds[0], bounds[1], dims)
                           for _ in range(pop_size)]
        # calculate initial intensity
        for firefly in self.population:
            firefly.update_intensity(self.benchmark)

    def step(self):
        # rank the solutions
        self.population.sort(key=operator.attrgetter('intensity'), reverse=True)
        # compare each pair of fireflies and move them accordingly
        for i in self.population:
            for j in self.population:
                if j.intensity > i.intensity:
                    i.move_firefly(j.position)
                    i.update_intensity(self.benchmark)
        # update the best solution
        if not self.best or self.population[0].intensity > self.best:
            self.best = self.population[0].intensity

    def run_optim(self):
        for it in range(self.max_iter):
            self.step()
        self.population.sort(key=operator.attrgetter('intensity'), reverse=True)
        return self.population[0].position, self.population[0].intensity

import TestFunction

fireflyOpt = FireflyOptimizer(TestFunction.four_peaks, [-5, 5], pop_size=30, dims=2, max_iters=100)
pos_best, intensity_best = fireflyOpt.run_optim()
print('best solution: ', pos_best)
print('best intensity: ', intensity_best)
