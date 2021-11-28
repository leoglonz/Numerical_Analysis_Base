import numpy as np
import numerics2_lonzarich as num

# Dependencies: LU_solve (see least squares)

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
    -> 'x_bar,' the n x 1 least squares solution vector that minimizes the Euclidean length of the residual b - Ax, where x is the so-called exat solution to Ax = b.
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
    x_bar = num.lusolve(AtA, At_b)

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
