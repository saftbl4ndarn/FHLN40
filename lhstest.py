import numpy as np
from pyDOE import lhs

# Define the number of dimensions for the mesh
dimensions = 2  # Example with 2 dimensions, you can change it as needed

# Define the number of points to generate
num_points = 25

# Generate the Latin Hypercube Sample
sample = lhs(dimensions, samples=num_points)

# Print the sample
print("Latin Hypercube Sample Points:")
print(sample)

# Define the lower and upper bounds for each dimension
lower_bounds = np.array([0, 0]) #[x1_low, x2_low]
upper_bounds = np.array([2, 1]) #[x1_upp, x2_upp]

# Scale the points

print("Scaled Latin Hypercube Sample Points:")
print(scaled_sample)
