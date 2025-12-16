import numpy as np
import matplotlib.pyplot as plt

np.random.seed(43)
T = 100 
mu = 0
sigma = 1.0

# --- Generate white noise --- 
w = np.random.normal(0, sigma, T)

# --- Generate realization --- 
x = np.zeros(T)
x[0] = 0
for t in range(1, T):
    x[t] = mu + x[t-1] + w[t]

# --- Differenced process --- 
y = x[1:] - x[:-1]

# --- Plot --- 
plt.figure(figsize=(10, 4))
plt.plot(x, label='$x_t$')
plt.plot(y, label='$\\nabla x_t$')
plt.xlabel('Time')
plt.legend()
plt.xlim(0, T-1)
plt.grid(True)
plt.tight_layout()
plt.savefig('ar1_process.pdf')
plt.close()
