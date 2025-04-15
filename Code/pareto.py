import numpy as np

# Given parameters
mu = 15  # Desired mean
alpha = 1.5  # Shape parameter (alpha > 1)

# Compute scale parameter (x_m)
x_m = mu * (alpha - 1) / alpha

# Generate Pareto random variables
rng = np.random.default_rng()  # Initialize NumPy random generator
# pareto_samples = x_m * (rng.pareto(alpha, size=5000)+1)  # Generate 1000 samples

# # Example: Print first 5 samples
# print(x_m)
# print(pareto_samples.mean())
n_samples = 50000
uniforms = rng.uniform(size=n_samples)
print(sum(np.where(uniforms>0.5,1,0))/n_samples)
