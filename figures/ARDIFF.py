import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
T = 100  # Number of time steps
mu = 0.5
sigma = 1.0

# Generate white noise
w = np.random.normal(0, sigma, T)

# AR(1) process (random walk with drift)
x = np.zeros(T)
x[0] = 0
for t in range(1, T):
    x[t] = mu + x[t-1] + w[t]

# Differenced process
y = x[1:] - x[:-1]

# Plot AR(1) process
plt.figure(figsize=(12, 6), dpi=150)
plt.plot(x, label='AR(1) process', color='navy', linewidth=2)
plt.title('AR(1) Process: $x_t = \\mu + x_{t-1} + w_t$', fontsize=16)
plt.xlabel('Time (t)', fontsize=14)
plt.ylabel('x_t', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ar1_process.png', format='png')
plt.close()

# Plot differenced process
plt.figure(figsize=(12, 6), dpi=150)
plt.plot(y, label='Differenced process', color='darkred', linewidth=2)
plt.title('Differenced Process: $\\nabla x_t = x_t - x_{t-1} = \\mu + w_t$', fontsize=16)
plt.xlabel('Time (t)', fontsize=14)
plt.ylabel('$\\nabla x_t$', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('differenced_process.png', format='png')
plt.close()