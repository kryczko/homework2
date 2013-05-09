"""

Same problem as before, just declaring a module to run the task.

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

def quad_interp(xi, yi):
    """ 

    This function computes the polynomial coefficients interpolated at given poi    nts for p(x)=c[0]+c[1]*x+c[2]*x**2

    """

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 3"
    assert len(xi)==3 and len(yi)==3, error_message

    # set up linear system to interpolate through data points

    A = np.vstack([np.ones(3), xi, xi**2]).T
    b = yi

    c = solve(A,b)
    return c

def plot_quad(xi,yi):
    
    quad_interp(xi,yi) 
    x = np.linspace(xi.min() - 1, xi.max() + 1, 1000)
    y = quad_interp(xi,yi)[0] + quad_interp(xi,yi)[1]*x + quad_interp(xi,yi)[2]*x**2
    plt.figure(1)
    plt.clf()
    plt.plot(x,y,'b-')

    plt.plot(xi, yi, 'ro')
    plt.ylim(-2, 8)

    plt.title("Data points and interpolating polynomial")

    plt.savefig('demo1plot.pdf')


def test_quad1():
    """

    Test code, no return value if test runs properly.

    """

    xi = np.array([-1., 0., 2.])
    yi = np.array([1., -1., 7.])

    c= quad_interp(xi,yi)
    c_true = np.array([-1., 0., 2.])

    print "c =     ", c
    print "c_true =     ", c_true

    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)

if __name__ == "__main__":
    print "Running test..."
    test_quad1()
