import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as stats

# --- Generate two datasets --- 
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=1000)
skewed_data = np.random.exponential(scale=1, size=1000)

# --- Plot --- 
plt.figure(figsize=(8, 4))

ax = plt.subplot(1, 2, 1)
qqplot(normal_data, line='s', ax=ax)
for line in ax.get_lines():
    mk = line.get_marker()
    if mk not in (None, 'None', ''):
        line.set_markersize(3)
ax.grid(True)
ax.set_title("")

ax = plt.subplot(1, 2, 2)
qqplot(skewed_data, line='s', ax=ax)
for line in ax.get_lines():
    mk = line.get_marker()
    if mk not in (None, 'None', ''):
        line.set_markersize(3)
        
ax.grid(True)
ax.set_title("")
plt.tight_layout()
plt.savefig("qq_plot_teori.pdf")
plt.close()
