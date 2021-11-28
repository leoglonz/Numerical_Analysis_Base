# Compute square roots of 3 and 5 up to 8 decimals using FPI


def g(x, a):
    return x - (x**2 - a)/6


def g3(x):
    return g(x, 3)


def g5(x):
    return g(x, 5)


# Find the sqrt of 3
tol = 1.0e-8
x_init = 5

rt, roots = fixedpt(g3, x_init, tol)
print(rt)

# Find the sqrt of 5
tol = 1.0e-8
x_init = 2.1

rt, roots = fixedpt(g5, x_init, tol)
print(rt)
