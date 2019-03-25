"""
Set up function to test the Firefly algorithm
"""

import math

def fourPeaks(x,y):
    term1 = math.e**( -(x-4)**2-(y-4)**2)
    term2 = math.e**(-(x+4)**2-(y-4)**2)
    term3 = 2 * (math.e**(-x**2-y**2) + math.e**(-x**2-(y+4)**2))
    return term1 + term2 + term3

if __name__ == '__main__':
    print(fourPeaks(4,4))
