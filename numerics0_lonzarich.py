
import numpy as np


def polynest(x, a, b=[]):
    # x is the input to the polynomial.
    # a is the vector of n+1 coefficients of the polynomial.
    # b is the vector of n base points. Default is empty.
    # This function evaluates a polynomial with coeficients a and basepoints b
    # at the point x and returns that value as y.

    a = np.asarray(a)
    b = np.asarray(b)
    adims = a.ndim
    bdims = b.ndim

    # in the case of error:
    if (adims != 1):
        raise ValueError('The coefficients (2nd input) must be passed as a vector (1 dimension).')

    if (adims != 1):
        raise ValueError('The coefficients (2nd input) must be passed as a vector (1 dimension).')

    n = a.size
    numbasepoints = b.size
    
    y = a[n-1]

    if numbasepoints:
        for idx in range(n-2, -1, -1):
            y = y*(x-b[idx])+a[idx]
    else:
        for idx in range(n-2, -1, -1):
            y = y*x + a[idx]

    return y
