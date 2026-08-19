def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

def second_derivative(f, x, h=1e-5):
    return (derivative(f, x + h, h) - derivative(f, x, h)) / h

def newton_optimizatio(f, x0):
    x = x0
    while True:
        first_derivative = derivative(f, x)
        second_derivative_value = second_derivative(f, x)

        x_new = x - first_derivative / second_derivative_value

        if abs(x_new - x) < 0.001:
            return x_new

        x = x_new