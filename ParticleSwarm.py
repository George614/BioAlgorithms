"""
Implementation of the Particle Swarm Optimization algorithm, both standard version and accelerated version
"""

import numpy as np
import random
import math

def func1(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total

class Particle:
    def __init__(self, bounds, dim):
        self.dim = dim
        self.position = np.random.uniform(bounds[0], bounds[1], self.dim)  # particle position
        self.velocity = np.zeros(self.dim)  # particle velocity
        self.pos_best = None  # best position individual
        self.err_best = -1  # best error individual
        self.error = -1       # error individual

    # evaluate current fitness
    def evaluate(self, obj_func):
        self.error = obj_func(self.position)
        # check to see if the current position is an individual best
        if self.error < self.err_best or self.err_best == -1:
            self.pos_best = self.position
            self.err_best = self.error

    # update new particle velocity
    def update_velocity(self, inertia=0.5, # constant inertia weight (how much to weigh the previous velocity)
                        beta = 2,  # cognitive constant
                        alpha = 2,  # social constant
                        pos_best_g=None):
        epsilon1 = np.random.rand()
        epsilon2 = np.random.rand()
        vel_cognitive = beta * epsilon2 * (self.pos_best - self.position)
        vel_social = alpha * epsilon1 * (pos_best_g - self.position)
        self.velocity = inertia * self.velocity + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position[i] = self.position[i] + self.velocity[i]

            # adjust maximum position if necessary
            if self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

            # adjust minimum position if neseccary
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]


class PSO:
    def __init__(self, obj_func, x0, bounds, num_particles, maxiter):
        global num_dimensions
        # x0 is initial position of
        num_dimensions = len(x0)
        err_best_g = -1  # best error for group
        pos_best_g = []  # best position for group

        # establish the swarm
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i = 0
        while i < maxiter:
            # print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, num_particles):
                swarm[j].evaluate(obj_func)

                # determine if current particle is the best (globally)
                if swarm[j].error < err_best_g or err_best_g == -1:
                    pos_best_g = list(swarm[j].position)
                    err_best_g = float(swarm[j].error)

            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i += 1

        # print final results
        print('FINAL:')
        print(pos_best_g)
        print(err_best_g)


initial = [5, 5]  # initial starting location [x1,x2...]
bounds = [(-10, 10), (-10, 10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
PSO(func1, initial, bounds, num_particles=15, maxiter=30)