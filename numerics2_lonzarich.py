import numpy as np


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

        # Stepping iteration counter and computing deviation between current
        # and previous approximations of x -- L2 norm of x^(k)-x^(k-1).
        iter += 1
        dev = (np.sum((x - x_old)**2))**0.5

        # Updating the previous estimate of x.
        x_old = x.copy()

    return x
