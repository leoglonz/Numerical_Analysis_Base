import numpy as np
import numerics0_lonzarich as num0
import numerics2_lonzarich as num2


# Dependencies: Jacobi (see spline), polynest (see divided differences)

''' ************* Newton's Divided Differences ************* '''
def newtondd(x_pts,y_pts):
    '''
    Overview:
        Using Newton's Divided Differences to find coefficients of
        interpolating polynomial.

    Inputs:
    -> 'x_pts' and 'y_pts' are both vectors that together represent the points
        (x[i], y[i]) that are being interpolated.

    Output:
    -> 'coeffs' is an array carrying the n+1 coefficients of the nth degree
        interpolating polynomial.
    '''
    #--------------------------------#

    # Number of points.
    n = len(x_pts)

    # Initializing NDD triangle, populating first column with y-values.
    triangle = np.zeros([n,n])
    triangle[::,0] = y_pts

    # Filling out triangle.
    for j in range(1,n):
        for i in range(n-j):
            triangle[i,j] = (triangle[i+1, j-1] - triangle[i, j-1])/ \
            (x_pts[i+j] - x_pts[i])

    # Capturing the first row, which contains all of the coefficients.
    coeffs = triangle[0]

    return coeffs


def newtonInterp(x_pts,y_pts):
    '''
    Overview:
        Use the coefficients collected by Newton's Divided Differences and
        construct a Lagrange interpolating polynomial.

    Inputs:
    -> 'x_pts' and 'y_pts' are both vectors that together represent the points
        (x[i], y[i]) that are being interpolated.

    Output:
    -> Returns 'Px,' the lowest degree polynomial that interpolates the given
        data points.
    '''
    #--------------------------------#

    # Retrieving polynomial coefficients via Newton DD.
    coeffs = newtondd(x_pts,y_pts)

    # Create callable function that represents lowest degree approximation.
    Px = lambda x: num0.polynest(x, coeffs, x_pts)

    return Px


''' ************* Chebyshev Interpolation ************* '''
def chebyshevRoots(numrts, interval = [-1, 1]):
    '''
    Overview:
        Calculates the n roots of the degree n Chebyshev polynomial T_n(x).

    Inputs:
    -> 'numrts,' the number of roots that are desired.
    -> 'interval,' the range of points relevant to interpolation. Shebyshev
    roots will be selected from this interval. Default is [-1, 1].

    Output:
    -> 'chebyRoots,' the n roots of the degree n Chebyshev polynomial.
    '''
    #--------------------------------#
    # Preallocating for efficiency.
    chebyRoots = np.zeros(numrts)

    # Calculating roots for standard interval.
    if (interval == [-1, 1]):
        for i in range(1, numrts+1):
            chebyRoots[i-1] = np.cos(((2*i - 1) * np.pi)/(2*numrts))

    # Throwing value error in the event that second input is not an interval.
    elif (isinstance(interval, list)) == False:
        raise ValueError('Invalid input interval. Select valid interval [a,b].')

    # Finding roots when the interval [-1, 1] is streched to some arbitrary
    # interval [a,b].
    else:
        a, b = min(interval), max(interval) # Assigning endpoints
        for i in range(1, numrts+1):
            chebyRoots[i-1] = (b+a)/2 + ((b-a)/2) * np.cos(((2*i - 1) * \
            np.pi)/(2*numrts))

    return chebyRoots


def chebyshevInterp(func, numrts, interval = [-1, 1]):
    '''
    Overview:
        Use n Chebyshev nodes to construct a Lagrange interpolating polynomial.

    Inputs:
    -> 'func,' the function being interpolated.
    -> 'numrts,' the number of roots that are desired, giving a degree numrts-1
    interpolating polynomial.
    -> 'interval,' the range of values from which Chebyshev nodes will be
    collected. The resulting interpolating polynomial is only relevant for this
    interval. Default is [-1, 1].

    Output:
    -> 'Px,' the degree numrts-1 Chebyshev interpolating polynomial constructed
    using Newton's Divided Differences.
    '''
    #--------------------------------#

    # Calculates the base points from the degree numrts Chebyshev polynomial.
    basepts = chebyshevRoots(numrts, interval)

    # Evaluating function at base points.
    ypts = func(basepts)

    # Using Newton's Divided Differences to construct the Chebyshev (Lagrange)
    # interpolating polynomial.
    Px = newtonInterp(basepts, ypts)

    return Px


''' ************* Cubic Spline Interp. Coefficients ************* '''
def cubiccoeff(x_pts, y_pts, tol = 1e-10, iterations = 1000):
    '''
    Overview:
        Computes the coefficients of a natural cubic spline fitting given
        points.

    Inputs:
    -> 'x_pts' and 'y_pts' are both vectors, together representing the
        points (x[i], y[i]) that are being interpolated.
    -> 'tol,' a tolerance that is passed for use in the Jacobi Method.
        Default is 1e-10.
    -> 'iterations,' the max number of iterations for the Jacobi Method.
        Default is 1000.

    Output:
    -> 'coeffs,' n-1 x n matrix, where the ith row contains coefficients a_i,
        b_i, c_i, and c_i of the ith equation of a cubic spline interpolating
        the given points.
    '''
    #--------------------------------#

    # Finding the number of data points, initializing n x n matrix A, n x 1
    # vector r, and n x 3 coefficient matrix.
    n = len(x_pts)

    A = np.zeros([n, n])
    r = np.zeros([n, 1])
    coeffs = np.zeros([n, 4]) # 1st column, all a_i; 2nd col, b_i; 3rd col, c_i;
                              # 4th column, all d_i.

    # Calculating deltas using discrete differences application from Numpy.
    delta_x = np.diff(x_pts)
    delta_y = np.diff(y_pts)

    # Setting natural spline endpoint conditions and filling out A with deltas
    # so that it is tridiagonal.
    A[0, 0] = 1
    A[-1, -1] = 1

    for i in range(1, n-1):
        A[i, i-1] = delta_x[i-1] # Diagonal left of center.
        A[i,i] = 2*(delta_x[i-1] + delta_x[i]) # Main diagonal.
        A[i, i+1] = delta_x[i]   # Diagonal right of center.

        # Also determining elements of r for Ac = r. Note that
        # r[0,0] = r[n,0] = 0.
        r[i,0] = 3*(delta_y[i]/delta_x[i] - delta_y[i-1]/delta_x[i-1])

    # Using the Jacobi Method to solve system of equations for the coefficients
    # c_i.
    c = num2.jacobi(A, r, np.zeros(len(A)), tol, iterations)


    # Finding the b_i and d_i coefficents using the c_i; Inserting all
    # coefficients into coefficient matrix.
    for i in range(0, n-1):
        coeffs[i,0] = y_pts[i] # Column for all a_i.
        coeffs[i,1] = (delta_y[i]/delta_x[i]) - (delta_x[i]/3)*(2*c[i] + \
        c[i+1]) # Column for all b_i.
        coeffs[i,2] = c[i] # Column for all c_i.
        coeffs[i,3] = (c[i+1] - c[i]) / (3*delta_x[i])  # Column for all d_i.

    # Returning n-1 x n matrix, where the ith row contains coefficients a_i,
    # b_i, c_i, and c_i of the ith spline.
    return coeffs[:-1,:]
