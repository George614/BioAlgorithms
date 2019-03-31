"""
Implementation of the Particle Swarm Optimization algorithm, both standard version and accelerated version
"""

import numpy as np

class Particle:
    def __init__(self, bounds, dim):
        self.dim = dim
        self.bounds = bounds
        self.position = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)  # particle position
        self.velocity = np.zeros(self.dim)  # particle velocity
        self.pos_best = None  # best individual position
        self.err_best = -1  # best individual error
        self.error = -1       # individual error at each iteration

    # evaluate current fitness
    def evaluate(self, obj_func):
        self.error = obj_func(self.position)
        # check to see if the current position is an individual best
        if self.error < self.err_best or self.err_best == -1:
            self.pos_best = self.position
            self.err_best = self.error

    # update new particle velocity
    def update_velocity(self, inertia=0.5,  # constant inertia weight (how much to weigh the previous velocity)
                        beta=2,   # cognitive constant
                        alpha=2,  # social constant
                        pos_best_g=None):
        epsilon1 = np.random.rand()
        epsilon2 = np.random.rand()
        vel_cognitive = beta * epsilon2 * (self.pos_best - self.position)
        vel_social = alpha * epsilon1 * (pos_best_g - self.position)
        self.velocity = inertia * self.velocity + vel_cognitive + vel_social

    # APSO update velocity
    def acc_update_velocity(self, beta=0.5, alpha0=0.5, it=0, gamma=0.95, pos_best_g=None):
        epsilon = np.random.normal()  # drawn from a Gaussian distribution
        alpha = alpha0 * np.exp(-gamma*it)
        vel_social = beta * (pos_best_g - self.position)
        vel_cognitive = alpha * epsilon
        self.velocity = self.velocity + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self):
        self.position = self.position + self.velocity
        # adjust position to be within the boundaries
        self.position[self.position > self.bounds[1]] = self.bounds[1]
        self.position[self.position < self.bounds[0]] = self.bounds[0]

class PSO:
    def __init__(self, obj_func, dimension, bounds, pop_size, max_iter, alpha,
                 acc=False, inertia=0.5, beta=2, gamma=0.95):
        err_best_g = -1  # best error for group
        pos_best_g = []  # best position for group
        # build the swarm
        swarm = []
        for i in range(pop_size):
            swarm.append(Particle(bounds, dimension))

        # begin optimization loop
        it = 0
        while it < max_iter:
            # evaluate fitness for each particle
            for j in range(pop_size):
                swarm[j].evaluate(obj_func)
                # determine if current particle is the global best
                if swarm[j].error < err_best_g or err_best_g == -1:
                    pos_best_g = list(swarm[j].position)
                    err_best_g = float(swarm[j].error)
            # update velocities and positions
            if acc:  # Use APSO
                for k  in range(pop_size):
                    swarm[k].acc_update_velocity(beta, alpha, it, gamma, pos_best_g)
                    swarm[k].update_position()
                it += 1
            else:   # Use PSO
                for k in range(pop_size):
                    swarm[k].update_velocity(inertia, beta, alpha, pos_best_g)
                    swarm[k].update_position()
                it += 1

        # print out results
        print("best position: ", pos_best_g)
        print("best error: ", err_best_g)

