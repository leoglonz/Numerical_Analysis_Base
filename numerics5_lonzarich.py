import numpy as np


''' ************* Simpson's Method ************* '''
def simpson(function, a, b):
    '''
    Overview:
        Here, we interpolate the function with a parabola that has been
        generically integrated. Conditions of the problem will then be
        substituted for a numerical solution.
    
    Inputs:
    -> 'function' is the curve we wish to integrate.
    -> 'a' and 'b' are the endpoints of the interval we will integrate through.
    
    Outputs:
    -> 'area,' the approximation of the integral of function over the given
        interval.
    '''
    #--------------------------------#
    
    # The midpoint of the interval:
    midpt = a + (b-a)/2
    
    # 'h' is the difference between the endpoint of the interval, 'b,' and its
    # midpoint.
    h = b - midpt
    
    # Simpson's Rule; we estimate the integral of our function using the
    # integral of a polynomial interpolant;
    # Functional evaluations give y0, y1, and y2, respectively.
    area = (h/3) * (function(a) + 4*function(midpt) + function(b))
    
    return area
    

''' ************* Composite Simpson's Method ************* '''
def compositeSimpson(function, n, a, b):
    '''
    Description:
        The approach of this implementation is to split the interval up into
        smaller subintervals and applying Simpson's Rule to each.

    Inputs:
    -> 'function' is the curve we wish to integrate;
    -> 'n' is the number of intervals we want integrate over; effectively, the
        number of steps;
    -> 'a' and 'b' are the endpoints of the interval we will integrate through.
    
    Outputs:
    -> 'area,' the approximation of the integral of function over the given
        interval.
    '''
    #--------------------------------#

    # Set of endpoints and midpoints defining each of n subintervals
    x = np.linspace(a,b,n+1)

    # The width of each subinterval.
    h = x[1]-x[0]

    # Creating a scaling array so that we can multiply through by f
    # evaluated at each point with its necessary scaler (i.e 1 for the
    # endpoints, 2 for the even midpoints, and 4 for the odd endpoints).
    # This will streamline the summation.
    c = 2*np.ones(n+1)
    c[0], c[-1] = 1.0, 1.0

    for i in range(1,len(c)-1):
        if i % 2 == 1:
            c[i] = 4

    # Computing area by summing over all scaled function evaluations.
    area = (h/3) * np.sum(c*function(x))

    return area


''' ************* Adaptive Quadrature (Simpson's) ************* '''
def adaptiveSimpson(fxn, a, b, tol=1e-12, Sab=0.0, recursion_counter=0):
    '''
    Description:
        Similar to composite Simpson's method, but here we The approach of this
        implementation is to subdivide intervals and apply the composite method
        to nested intervals until the total error reaches a set ceiling.

    Inputs:
    -> 'fxn' is the function we are integrating.
    -> 'a' and 'b' are the endpoints of the interval we will integrate across.
    -> Stopping Conditions: 'tol' gives the maximum error we are willing to
       accept, and 'recursion_counter' counts number of recursions (cap at 20).
    -> Sab is the value of Simpson's Method on the interval of [a,b] we are
       currently working on.

    Outputs:
    -> Total sum, 's_ac + s_cb' over specified interval, or recursive
        subdivisions
        
        
    ** 'S_ab' and 'recursion_counter' are only passed internally **
    '''
    #--------------------------------#

    # Finding midpoint of the current interval [a,b].
    c = (a+b)/2

    # Find s_ab, sum over the current interval, [a,b], using Simpson's Method...
    s_ab = simpson(fxn, a, b)

    # ... And for the intervals [a, c] and [c, b], where 'c' is the midpoint.
    s_ac = simpson(fxn, a, c)

    s_cb = simpson(fxn, c, b)

    # Throwing value error if maximum recursion depth is reached;
    # Max = 20 since that has the potential produce 2^20 (about a million)
    # function calls.
    if recursion_counter == 20:
        raise ValueError('Recursion depth too low. Either choose larger error \
        tolerance or set lower depth.')

    # Checking devation of sums of subintervals from sum over entire interval.
    if (np.abs(s_ab - s_ac - s_cb) < 10*tol):
        return s_ac + s_cb       # If within tolerance, accept the current sum.
    else:
        return adaptiveSimpson(fxn, c, b, tol/2, s_cb, recursion_counter+1) +\
        adaptiveSimpson(fxn, a, c, tol/2, s_ac, recursion_counter+1)


''' ************* Gaussian Quadrature ************* '''
def gaussQuad(function, n, a=-1, b=1):
    '''
    Description:

    Inputs:
    -> 'function' is the function we are integrating.
    -> 'n' is the number of nodes (roots of Legendre polynomial p_n(x)).
    -> 'a' and 'b' are the endpoints of the interval we will integrate across.
        -1 and 1 are the defaults.

    Output:
    -> 'result' is the value of the integral of 'function' on the
        interval [a,b].
    '''
    #--------------------------------#

    # Finding the roots and weights: Passing our degree, n, outputs sample
    # nodes and corresponding weights 'wts.'
    nodes, wts = np.polynomial.legendre.leggauss(n)

    # Since Gaussian Quadrature was developed for the interval [-1.1], we
    # perform a transformation to [a,b]. (Note that this changes nothing if
    # [a,b] = [-1,1].)
    midpt = (a+b)/2.0
    halfWidth = (b-a)/2.0
    nodes = halfWidth*nodes + midpt

    # Evaluating and storing function evaluations for nodes.
    fNodes = function(nodes)

    # Computing the inner product.
    result = np.dot(wts, fNodes)

    # Multiply by the halflength for the interval transformation. (If no
    # transformation, 'result*halflength = result.')
    return result*halfWidth
