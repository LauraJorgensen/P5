import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Generate two datasets
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=1000)   # normal distribution
skewed_data = np.random.exponential(scale=1, size=1000)     # skewed distribution

# 2. Create Q-Q plots
plt.figure(figsize=(10, 5))

# Q-Q plot for normal data
plt.subplot(1, 2, 1)
stats.probplot(normal_data, dist="norm", plot=plt)
plt.title("Q-Q Plot: Normal Data")

# Q-Q plot for skewed data
plt.subplot(1, 2, 2)
stats.probplot(skewed_data, dist="norm", plot=plt)
plt.title("Q-Q Plot: Skewed Data")

plt.tight_layout()
plt.show()