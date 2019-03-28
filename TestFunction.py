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

