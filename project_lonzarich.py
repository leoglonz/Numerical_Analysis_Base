import numpy as np


''' ************* Vectorized Naive DFT ************* '''
def dft(x):
    '''
    Description:
        Computs the discrete Fourier transform of a given vector, x.

    Inputs:
    -> 'x', a vector of n uniformly sampled components.

    Output:
    -> 'y,' The n-component DFT of x
    '''
    #--------------------------------#

    # Ensuring x is array-like and finding the number of elements.
    x = np.asarray(x, dtype=float)
    n = len(x)
    
    # Creating arrays such that l x k gives the n x n matrix M for the DFT.
    l = np.arange(n)
    k = l.reshape((n, 1))
    
    
    M = np.exp(-2j * np.pi * k * l / n)
    
    # Computing the DFT of x.
    y = (1/np.sqrt(n)) * np.dot(M, x)
    
    return y


''' ************* Non-vectorized Naive DFT ************* '''
'''
def nonvec_dft(x):
    n = len(x)
    y = np.zeros(n, dtype = complex)
    for k in range(n):  # For each output element
        s = complex(0)
        for l in range(n):  # For each input element
            angle = 2j * np.pi * l * k / n
            s += x[l] * np.exp(-angle)
        y[k] = s
    return (1/np.sqrt(n)) * y
'''

