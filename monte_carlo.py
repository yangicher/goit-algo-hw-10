import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# -------------------------
# Part 1: Plotting the Function
# -------------------------

# Define the function to integrate: f(x) = x^2
def f(x):
    return x ** 2

# Integration limits
a = 0  # lower bound
b = 2  # upper bound

# Create an array of x values for plotting
x = np.linspace(-0.5, 2.5, 400)
y = f(x)

# Create the plot
fig, ax = plt.subplots()

# Plot the function f(x)
ax.plot(x, y, 'r', linewidth=2, label='$f(x)=x^2$')

# Fill the area under the curve between a and b
ix = np.linspace(a, b, 200)
iy = f(ix)
ax.fill_between(ix, iy, color='gray', alpha=0.3, label='Area under f(x)')

# Set plot limits and labels
ax.set_xlim([x[0], x[-1]])
ax.set_ylim([0, max(y) + 0.5])
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.axvline(x=a, color='gray', linestyle='--')
ax.axvline(x=b, color='gray', linestyle='--')
ax.set_title('Integration of $f(x)=x^2$ from ' + str(a) + ' to ' + str(b))
ax.legend()
plt.grid()
plt.show()

# -------------------------
# Part 2: Monte Carlo Integration
# -------------------------

# Monte Carlo Integration parameters
N = 1000000  # number of random points

# The bounding rectangle:
# x in [a, b] and y in [0, f(b)] because f(x)=x^2 is increasing in [0,2]
x_random = np.random.uniform(a, b, N)
y_random = np.random.uniform(0, f(b), N)

# Count how many random points fall under the curve f(x)
points_under_curve = y_random <= f(x_random)
num_under_curve = np.sum(points_under_curve)

# The area of the rectangle that encloses the curve:
rect_area = (b - a) * f(b)

# Monte Carlo estimate of the integral (area under the curve)
monte_carlo_area = (num_under_curve / N) * rect_area

print("Monte Carlo estimated integral (area under the curve):", monte_carlo_area)

# -------------------------
# Part 3: Verification using SciPy's quad
# -------------------------

# Use quad to compute the integral analytically
quad_result, quad_error = spi.quad(f, a, b)
print("SciPy quad integration result:", quad_result)
print("Estimated error from quad:", quad_error)