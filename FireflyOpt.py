"""
Find optima of a given function using firefly algorithm
"""

from Firefly import Firefly
import numpy as np

class FireflyOptimizer:
    def __init__(self, **kwargs):
        self.population_size = int(kwargs.get('population_size', 10))
        self.problem_dim = kwargs.get('problem_dim', 2)
        self.lower_bound = kwargs.get('lower_bound', -5)
        self.upper_bound = kwargs.get('upper_bound', 5)
        self.generations = kwargs.get('generations', 50)
        self.gamma = kwargs.get('gamma', 0.97)  # absorption coefficient
        self.alpha = kwargs.get('alpha', 0.25)  # randomness [0,1]
        self.beta_0 = kwargs.get('beta_0', 1)   # attractiveness at distance=0
        self.benchmark = kwargs.get('benchmark', None)  # this is the function to be optimized
        self.best = None
        self.population = [Firefly(self.alpha,self.beta_0,self.gamma,self.lower_bound,self.upper_bound,self.problem_dim)
                           for _ in range(self.population_size)]
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
