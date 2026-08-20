import numpy as np
from scipy.differentiate import jacobian, hessian

def derivative(f, x, h=1e-5):
    """:D Approximate the first derivative of f at x using forward difference."""
    return (f(x + h) - f(x)) / h


def second_derivative(f, x, h=1e-5):
    """Approximate the second derivative of f at x."""
    return (derivative(f, x + h, h) - derivative(f, x, h)) / h


def optimize(f, x0, h=1e-5, max_iter=100000):
    """Find a local minimum of f using Newton's method starting at x0.
    Use methods above to compute the first derivative and second derivative.

    Parameters:
    f -- function to be optimized
    x0 -- initial value to start with
    h -- value used to compute derivatives, default set to 1e-5
    max_iter -- maximum number of iterations to avoid inifinite loop

    Returns:
    Estimated location of a local minimum.
    """

    if not callable(f):
        raise TypeError("f must be a function")
        
    if not isinstance(x0, (int, float)):
        raise TypeError("x0 must be numeric")
    
    x = x0
    for _ in range(max_iter):
        first_derivative = derivative(f, x)
        second_derivative_value = second_derivative(f, x)

        x_new = x - first_derivative / second_derivative_value

        if abs(x_new - x) < 0.00001:
            return x_new

        x = x_new

    return x

    def optimize_multivariate(f, x0, tol=1e-5, max_iter=100000):
        """
        Find a local minimum of a multivariate function using Newton's method.

        Parameters
        ----------
        f : function
            Scalar-valued function to be optimized.
        x0 : array-like
            Initial starting point.
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        numpy.ndarray
            Estimated location of a local minimum.
        """
        if not callable(f):
            raise TypeError("f must be a function")
        
        if not isinstance(x0, (int, float)):
            raise TypeError("x0 must be numeric")

        x = np.array(x0, dtype=float)
        
        for _ in range(max_iter):
            # gradient vector
            g = jacobian(f, x).df

            # Hessian matrix
            H = hessian(f, x).ddf

            # Solve H * step = g
            step = np.linalg.solve(H, g)

            # Newton update
            x_new = x - step

            # stopping condition
            if np.linalg.norm(x_new - x) < tol:
                return x_new

            x = x_new

        return x