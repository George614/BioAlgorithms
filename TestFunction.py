"""
Set up function to test the Firefly algorithm
"""

import math
import numpy as np

# example function used in chapter 8
def four_peaks(x):
    term1 = math.e**(-(x[0]-4)**2-(x[1]-4)**2)
    term2 = math.e**(-(x[0]+4)**2-(x[1]-4)**2)
    term3 = 2 * (math.e**(-x[0]**2-x[1]**2) + math.e**(-x[0]**2-(x[1]+4)**2))
    return term1 + term2 + term3

# function 28 in the book
def egg_create(x):
    return x[0]**2 + x[1]**2 + 25*(math.sin(x[0])**2 + math.sin(x[1])**2)
# function 30 in the book
def exponential_(x):
    return -np.exp(-0.5 * np.sum(np.square(x)))

def ackley(x):
    dim = x.size
    return -20 * np.exp(-0.02*np.sqrt(1/dim * np.sum(np.square(x))))-np.exp(1/dim * np.sum(np.cos(2*np.pi*x)))+20+np.e

def easom(x):
    dim = x.size
    return (-1)**(dim+1)*np.prod(np.cos(x)*np.exp(-1*np.sum(np.square(x-np.pi))))

def rosenbrock(x):
    y = 0
    for i in range(x.size-1):
        y += (x[i]-1)**2 + 100*(x[i+1]-x[i]**2)**2
    return y

# if __name__ == '__main__':
    test_a = ackley(np.zeros((7,)))
    print(test_a)
    test_e = easom(np.ones((6,))*np.pi)
    print(test_e)
    test_r = rosenbrock(np.ones((6,)))
    print(test_r)
