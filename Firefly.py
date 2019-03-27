"""
Firefly algorithm implementation
"""

import numpy as np
from numpy import linalg as LA
import random


class Firefly:
    def __init__(self, alpha, beta, gamma, lower_bound, upper_bound, problem_dimension):
        self.alpha = alpha        # control of randomness
        self.beta = beta          # attractiveness
        self.gamma = gamma        # light absorption coefficient
        self.intensity = None     # Intensity of individual firefly
        self.lower_bound = lower_bound         # lower boundary of the searching space we consider (single dim)
        self.upper_bound = upper_bound         # upper boundary of the searching space we consider (single dim)
        self.position = np.array([random.uniform(self.__lower_boundary, self.__upper_boundary)
                                    for _ in range(problem_dimension)])

    def move_firefly(self, j_position):
        # Cartesian distance
        dist = LA.norm(self.position - j_position)
        # update position
        self.position = self.position + self.beta*np.exp(-self.gamma*dist**2) * (j_position-self.position) + \
            self.alpha * (random.uniform(0, 1)-0.5)
        # clamp the value in the boundaries
        for idx, position in np.ndenumerate(self.position):
            if position < self.lower_bound:
                self.position[idx] = self.lower_bound
            elif position > self.upper_bound:
                self.position[idx] = self.upper_bound

    def update_intensity(self,obj_func):
        self.intensity = obj_func(self.position)