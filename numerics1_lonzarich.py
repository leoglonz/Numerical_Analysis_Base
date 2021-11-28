import numpy as np


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
