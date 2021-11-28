import numpy as np


''' ____________************* Numerics 0 *************____________ '''

''' ************* Polynomial Nesting ************* '''
def polynest(x, a, b=[]):
    # x is the input to the polynomial.
    # a is the vector of n+1 coefficients of the polynomial.
    # b is the vector of n base points. Default is empty.
    # This function evaluates a polynomial with coeficients a and
    # basepoints b
    # at the point x and returns that value as y.

    a = np.asarray(a)
    b = np.asarray(b)
    adims = a.ndim
    bdims = b.ndim

    # in the case of error:
    if (adims != 1):
        raise ValueError('The coefficients (2nd input) must be ' \
        'passed as a vector (1 dimension).')

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




''' ____________************* Numerics 1 *************____________ '''

''' ************* Bisection ************* '''
def bisection(fxn, ainit, binit, tol = 0.000001, maxIter = 250):
    # fxn is the function that we are trying to determine the root for.
    # The interval containing a root is [ainit, binit].
    # Optional arguments of a tolerance (default 10^(-6)) and maximum
    # iterations (default 250).
    # This function will return an approximation to a root of fxn
    # inside [ainit, binit] within tol.

    # Evaluate the sign of the function at the endpoints.
    fa = np.sign(fxn(ainit))
    fb = np.sign(fxn(binit))

    # Check if this initial interval is guaranteed to contain a root.
    if (fa*fb >= 0):
        if (fa == 0):
            return ainit
        elif (fb == 0):
            return binit
        else:
            raise ValueError('The input interval does not have opposite sign' \
            ' of function values at the endpoints. The interval may not ' \
            'contain a root.')
    else:
        # Read the input interval into local variables
        a = ainit
        b = binit
        err = b-a
        itr = 0
        midpts = np.zeros((maxIter,1))

        while ((err > tol) and (itr < maxIter)):
            # The interval length will be cut in half
            err = 0.5*err
            # The midpoint is now half the interval from the left endpoint
            c = a + err
            # Recoord this midpoint (iteration, approximation) in a vector
            midpts[itr] = c
            # Evaluate the function at the midpoint
            fc = np.sign(fxn(c))

            # Select a half interval with a root in it
            if (fa*fc < 0):
                b = c
                fb = fc
            else:
                a = c
                fa = fc

            # Now we need to update the iteration counter
            itr += 1

        # We satisfied out stopping criteria, so we return the midpoint of
        # the last step.
        # Let us record the number of iterations and only return the recorded
        # values in midpts
    return c, midpts[:itr]


''' ************* Fixed Point Iteration ************* '''
def fixedpt(gxn, xinit, tol = 1.0e-6, maxIter = 250):
    # gxn is the function we seek to determine the fixed points of.
    # xinit is our initial estimate for the fixed point iteration.
    # Optional arguments of a tolerance (default 10^(-6)) and maximum
    # iterations (default 250).
    # This function will return an approximation of a fixed point of gxn
    # within tol and the maximum number of iterations.

    # Read the input interval into local variables
    r = xinit
    err = np.abs(r)
    itr = 1
    roots = np.zeros((maxIter,1))

    while ((err > tol) and (itr < maxIter)):
        # Computing error at step itr.
        err = np.abs(gxn(r) - r)
        # Next approximation of the fixed point/root.
        r = gxn(r)
        # Collecting each iteration's estimation of a root of gxn.
        roots[itr] = r

        # Updating the iteration counter.
        itr += 1

    # We satisfied our conditions, and so we return the best approximation of the
    # fixed point along with a convergent sequence (roots) of approximations.
    return r, roots[:itr]


''' ************* Newton's Method ************* '''
def newton(fxn, dfxn, xinit, tol = 1.0e-6, maxIter = 250):
    # fxn is the function we seek to determine the fixed points of.
    # dfxn is the derivitive of fxn.
    # xinit is our initial estimate for the fixed point iteration.
    # Optional arguments of a tolerance (default 10^(-6)) and maximum
    # iterations (default 250).
    # This function will return an approximation of a fixed point of gxb
    # inside within tol and maximum number of iterations.

    # Read the input interval into local variables
    r = xinit
    err = np.abs(r)
    itr = 1
    roots = np.zeros((maxIter, 1))

    while ((err > tol) and (itr < maxIter)):
        # Computing error at step itr.
        err = np.abs(fxn(r)/dfxn(r))
        # Next approximation of the fixed point/root.
        r += - fxn(r)/dfxn(r)
        # Collecting each iteration's estimation of a root of gxn.
        roots[itr] = r

        # Updating the iteration counter.
        itr += 1

    # We satisfied our conditions, and so we return the best approximation of
    # the fixed point along with a convergent sequence (roots) of
    # approximations.
    return r, roots[:itr]
    



''' ____________************* Numerics 2 *************____________ '''

''' ************* Row Operations ************* '''
def rowdiff(Mat, k, j, scaling):
    # read row k, scale row j, and then subtract row j from row k and store in
    # row k
    Mat[k, :] = Mat[k, :] - scaling*Mat[j, :]
    return Mat


# Row swapping
def rowswap(Mat, k, j):
    Mat[[k, j], :] = Mat[[j, k], :]
    return Mat


# Row scaling
def rowscale(Mat, k, scaling):
    Mat[k, :] = scaling*Mat[k, :]
    return Mat


def lu(Mat):
    # Here we take a matrix Mat and perform a decomposition such that we return
    # the factorization L, U, and P satisfying P * Mat = L * U.

    # Finding dimensions of A.
    A = np.asarray(Mat)
    m, n = len(A), len(A[0])

    # We only want this for square matrices.
    if (m != n):
        raise ValueError('The input matrix must be square; Your matrix has \
        size %d x %d.' % (m, n))

    # Preallocate n x n matrices for lower-triangular matrix L, and pivot, P.
    L = np.zeros([m, n])
    P = np.identity(n)

    # We now cycle through the first n-1 columns with pivoting and elimination.
    for col in range(n):
        # Find row with the largest magnitude entry in this column on or below
        # the diagonal:
        pivot_index = np.argmax(abs(A[col:n, col]))
        # Knowing that P is the index in the diagonal or below, shift by i-1.
        pivot_index += col
        # The value of this absolute argmax becomes the pivot
        pivot = np.float(A[pivot_index, col])
        # If pivot is zero, the matrix is singular. If it is not, row swap to
        # obtain the preferred pivot
        if (pivot):
            if (pivot_index != col):
                # This means the pivot is nonzero; swap A, the permutation
                # matrix P, and the current iteration of L
                A = rowswap(A, pivot_index, col)
                L = rowswap(L, pivot_index, col)
                P = rowswap(P, pivot_index, col)
        else:
            raise ValueError('The matrix provided is singular so' \
            ' decomposition fails.')

        # For each row index greater than the column index, we need to do
        # the rowcombo with scalefactor the ratio of the entries in the
        # respective column.
        for row in range(col+1, n):
            scalefactor = A[row,col]/pivot
            L[row, col] = scalefactor
            A = rowdiff(A, row, col, scalefactor)

    # With the row swapping, we must include the ones along the diagonal
    L += np.identity(n)

    # Return L, P, and A since A is now upper triangular.
    return L, A, P


def backsub(U, b):
    # Takes an n x n upper-triangular matrix U and solves the system Ux = b
    # using backward substitution, returning the n x 1 solution vector x.

    # Ensuring input is array-like.
    U = np.asarray(U)
    b = np.asarray(b)
    # Get sizes of U and b, prreallocating solution vector.
    n = len(b)
    x = np.zeros(n)
    (mU, nU) = U.shape

    # Confirming dimensions match for back-substitiution.
    if (mU != nU) or (n != nU):
        raise ValueError('The dimensions of the input are not correct. U must \
        be square and b a vector. The number of columns in U must match the l \
        length of b.')


    #Reading last entry in.
    x[n-1] = b[n-1]/np.float(U[n-1, n-1])

    # Working backwards from row n-2 to row 0
    for i in range(n-2, -1, -1):
        # We find the inner product, subtract it from the corresponding value
        # in b, and divide by the corresponding diagonal element in U.
        x[i] = (b[i] - np.dot(U[i, i + 1:n], x[i+1:n]))/U[i, i]

    return x


def forwardsub(L, b):
    # Takes a lower triangular n x n matrix L and solves the system Lx = b
    # using forwards substitution and returns the n x 1 solution x.

    # Ensure that inputs are array-like.
    L = np.asarray(L)
    b = np.asarray(b)

    # Getting sizes and shapes of inputs.
    n = b.size
    (mL, nL) = L.shape

    # Ensure they are compatible for forward substitiution.
    if (mL != nL) or (n != nL):
        raise ValueError('The dimensions of the input are not correct. L' \
        ' must be square and b a vector. The number of columns in L must' \
        ' match the length of b.')

    # Preallocating output solution vector.
    x = np.zeros(b.shape)

    # Reading first entry into place.
    x[0] = b[0]/np.float(L[0, 0])

    # Work from row 1 up to row n-1.
    for i in range(1,n):
        # We have found the values of x with smaller index than the current row.
        # the lower triangular matrix may have factors for those values of x.
        # We complete the short inner product, subtract it from the associated
        # value in b, and divide by the diagonal element in L.
        # (This function performs the division by L[row,row] for generality.)
        x[i] = (b[i] - np.dot(L[i, 0:i], x[0:i]))/L[i, i]

    return x


def fbsolve(L, U, b):
    # Conducts a forward/backward substitution to solve the system LUx = b.

    # Ensuring inputs are array-like.
    L = np.asarray(L)
    U = np.asarray(U)
    b = np.asarray(b)
    n = b.size
    (mU, nU) = U.shape
    (mL, nL) = L.shape

    # Confirming dimensions match for forward substitiution.
    if (mL != nL) or (n != nL) or (mU != nU) or (nU != n):
        raise ValueError('The dimensions of the input are not correct. U and L \
        must be square and be a vector. The number of columns in U and L must \
        match the length of b.')

    # Executing forward and backward substitiutions.
    y = forwardsub(L, b)
    x = backsub(U, y)

    return x


''' ************* Complete LU Solve ************* '''
def lusolve(A, b):
    # Solves the system equation Ax = b and returns x by first executing an LU
    # factorization, and then running forward and backward substitiutions.

    P, L, U = lu(A)
    #U = np.array([[15, 55],[ 0, -3.3]])
    x = fbsolve(L, U, np.dot(P, b))

    return x


''' ************* Jacobi Method ************* '''
def jacobi(A, b, xinit, tol = 1e-10, maxIter = 1000):
    '''
    Overview:
        Uses the Jacobi Method to iteratively solve the linear system of
        equations Ax = b.

    Inputs:
    -> 'A' is an n x n coefficient matrix where each row represents the
        coefficients of one equation in the system.
    -> 'b' is an n x 1 vector containing the right-hand-side values of each
        equation.
    -> 'tol,' a tolerance that places an upper limit on the amount of error
        acceptable between the current and previous approximations of x.
        Default is 1e-10.
    -> 'maxIter' sets a cap on the maximum number of fixed point iterations
        so that the computation does not get unwieldy. Default is set to 1000.

    Output:
    -> 'x,' the n x 1 vector that gives approximated solutions x_i to the
        linear system.
    '''
    #--------------------------------#

    # Finding the number of equations.
    n = len(A)

    # Initializing prior estimate of x and iteration count
    xinit = np.asarray(xinit)
    x_old = xinit.copy()
    x = xinit.copy()
    A = np.array(A) # Eunsuring A is an array.

    # Setting iterations to zero, and giving a placeholder to begin iterative.
    iter = 0
    dev = 1 + tol

    # Calculating the kth iteration of x using the k-1th approximation until
    # stopping conditions are met.
    while ((dev > tol) and (iter < maxIter)):
        for i in range(0, n):
            # Resetting sum because it only applies to the ith component of x.
            sum = 0
            for j in range(0, n):
                if j != i:
                    sum += A[i, j] * x_old[j]
            # New estimate of ith component of x.
            x[i] = (b[i] - sum)/A[i,i]

        # Stepping iteration counter and computing deviation between current
        # and previous approximations of x -- L2 norm of x^(k)-x^(k-1).
        iter += 1
        dev = (np.sum((x - x_old)**2))**0.5

        # Updating the previous estimate of x.
        x_old = x.copy()

    return x


''' ************* Gauss-Seidel Method ************* '''
def gausssiedel(A, b, xinit, tol = 1e-10, maxIter = 1000):
    '''
    Overview:
        Uses the Gauss-Seidel Method to iteratively solve the linear system of
        equations Ax = b.

    Inputs:
    -> 'A' is an n x n coefficient matrix where each row represents the
        coefficients of one equation in the system.
    -> 'b' is an n x 1 vector containing the right-hand-side values of each
        equation.
    -> 'tol,' a tolerance that places an upper limit on the amount of error
        acceptable between the current and previous approximations of x.
        Default is 1e-10.
    -> 'maxIter' sets a cap on the maximum number of fixed point iterations
        so that the computation does not get unwieldy. Default is set to 1000.

    Output:
    -> 'x,' the n x 1 vector that gives approximated solutions x_i to the
        linear system.
    '''
    #--------------------------------#

    # Finding the number of equations.
    n = len(A)

    # Initializing prior estimate of x and iteration count
    xinit = np.asarray(xinit)
    x_old = xinit.copy()
    x = xinit.copy()
    A = np.array(A) # Eunsuring A is an array.

    # Setting iterations to zero, and giving a placeholder to begin iterative.
    iter = 0
    dev = 1 + tol

    # Calculating the kth iteration of x using previous and current
    # approximations until stopping conditions are met.
    while ((dev > tol) and (iter < maxIter)):
        for i in range(0, n):
            # Resetting sums because they only apply to the ith component of x.
            sum1 = 0
            sum2 = 0
            for j in range(0, i):
                    sum1 += A[i, j] * x[j]

            for j in range(i + 1, n):
                    sum2 += A[i, j] * x_old[j]

            # New estimate of ith component of x.
            x[i] = (b[i] - sum1 - sum2)/A[i,i]

        # Stepping iteration counter and computing deviation between
        # current
        # and previous approximations of x -- L2 norm of x^(k)-x^(k-1).
        iter += 1
        dev = (np.sum((x - x_old)**2))**0.5

        # Updating the previous estimate of x.
        x_old = x.copy()

    return x




''' ____________************* Numerics 4 *************____________ '''

''' ************* Least Squares (w/PALU) ************* '''
def leastSquares_lu(A, b):
    '''
    Overview:
        Uses PALU factorization to find a least squares solution, x_bar, to an
        inconsistent system of equations, Ax = b.

    Inputs:
    -> 'A' is an m x n coefficient matrix where each row represents the
        coefficients of one equation in the system.
    -> 'b' is an n x 1 vector containing the right-hand-side values of each
        equation.

    Output:
    -> 'x_bar,' the n x 1 least squares solution vector that minimizes the
    Euclidean length of the residual b - Ax, where x is the so-called exact
    solution to Ax = b.
    '''
    #--------------------------------#

    # Ensuring A, b are array-like and calculating transpose of A.
    A = np.array(A)
    b = np.array(b)
    At = np.transpose(A)

    # Performing matrix multiplications A^T A and  A^T b to form the system of
    # normal equations.
    AtA = np.dot(At,A)
    At_b = np.dot(At,b)

    # Finding the least squares solution with PALU factorization.
    x_bar = lusolve(AtA, At_b)

    # Calculating the residual.
    r = b - np.dot(A, x_bar)

    return x_bar, r


''' ************* QR Factorization (w/Gram-Schmidt) ************* '''
def qr(A):
    '''
    Overview:
        Computes the full QR factorization of a matrix A using by applying
        classical Gram-Schmidt orthogonalization to the columns of A.

    Inputs:
    -> 'A,' an m x n matrix.

    Output:
    -> 'Q' is a square m x m matrix
    -> 'R,' an m x n upper triangular matrix that is the same shape as A.
    '''
    #--------------------------------#

    # Ensuring that A is array-like, and capturing its dimensions.
    A = np.asarray(A)
    (m, n) = np.shape(A)

    # Preallocating matrices for R and Q.
    R = np.zeros([m, n])
    Q = np.zeros([m, m])

    # Running through each column of A (call them y).
    for j in range(n):
        y = A[:, j]

        for i in range(j):
            q = Q[:,i]
            # Filling out the upper-triangular section of R with the inner
            # products.
            R[i, j] = np.dot(np.transpose(q), A[:,j])
            y = y - R[i, j] * q

        # Filling out the diagonals with the magnitudes of vector y.
        R[j, j] = np.linalg.norm(y)

        # Calculating the unit vector.
        Q[:,j] = y/R[j, j]

    # Returning orthogonal matrix Q and upper-triangular matrix R that form the
    # QR factorization of A.
    return Q, R


''' ************* Least Squares (w/QR Factorization) ************* '''
def qrsolve(A,b):
    '''
    Overview:
        Uses the QR factorization of a matrix A to find the least squares
        solution to the inconsistent system Ax = b.

    Inputs:
    -> 'A' is the m x n coefficient matrix of an inconsistent system.
    -> 'b' is an n x 1 vector containing the right-hand-side values of each
        equation in the system.

    Output:
    -> 'Q' is a square m x m matrix
    -> 'R,' an m x n upper triangular matrix that is the same shape as A.
    '''
    #--------------------------------#

    # Ensuring that A and b are array-like, and capturing the dimension of A.
    A = np.asarray(A)
    b = np.asarray(b)
    (m, n) = np.shape(A)

    # Finding Q and R using Gram-Schmidt orthogonalization process.
    Q, R = qr(A)

    # Calculating right-hand side of system Rx = Q^Tb from Ax=QRx=b.
    d = np.dot(np.transpose(Q), b)

    # Selecting the first n rows and columns from R, and first n rows from d.
    R_redux = R[0:n, 0:n]
    d_redux = d[0:n]

    # Finding solution set x_bar for Ax = b that minimizes the L2-norm of the
    # residuals. Also returning the set of residuals.
    x_bar, resid = leastSquares_lu(R_redux, d_redux)

    # Returning the solution set x_bar for the inconsistentsystem of equations
    # Ax = b, along with the set of residuals.
    return x_bar, resid




''' ____________************* Numerics 3 *************____________ '''

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
    Px = lambda x: polynest(x, coeffs, x_pts)

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
    c = jacobi(A, r, np.zeros(len(A)), tol, iterations)


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




''' ____________************* Numerics 5 *************____________ '''

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
