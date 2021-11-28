import numpy as np
import matplotlib.pyplot as plt
import numerics1_lonzarich as num


def fxn(x):
  return x**2 - 0.24-x
ainput = 1.0  # left endpoint
binput = 2.0  # and right endpoint of the interval for bisection.
xinit  = 1.4  # this is the inital estimate for fixed point iteration and newton


tol = 1.0e-10
rt, roots = num.bisection(fxn, ainput, binput, tol)
print(rt, roots.size)
