"""

Homework due for the third week of Coursera's Scientific computing course.

"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

# Data points
xi = np.array([-1., 1., 2.])
yi = np.array([0., 4., 3.])

#Main Matrix and what the matrix is equal to
A = np.array([[1., -1., 1.], [1., 1., 1.], [1., 2., 4.]])
b = yi

# Solving it...
c = solve(A,b)

print "The polynomial coefficients are:"
print c

#Plot the resulting polynomial
x = np.linspace(-2,3,1001)
y = c[0] + c[1]*x +c[2]*x**2

plt.figure(1)
plt.clf()
plt.plot(x,y,'b-')

plt.plot(xi, yi, 'ro')
plt.ylim(-2, 8)

plt.title("Data points and interpolating polynomial")

plt.savefig('demo1plot.pdf')




