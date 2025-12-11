import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.tsa.arima_process import arma_acf, arma_pacf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima_process import ArmaProcess, arma_acf, arma_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

# Generer syntetisk ARMA data og estimerer ARMA model på denne

np.random.seed(43)

# Test proces: ARMA(1,1)
ar_true = np.array([1, -0.6])   # AR: 1 - 0.6 L
ma_true = np.array([1, 0.5])    # MA: 1 + 0.5 L

arma = ArmaProcess(ar_true, ma_true)
y = pd.Series(arma.generate_sample(nsample=365, burnin=200))

train = y

# find bedste ARMA(p,d,q) model via AIC
best_aic = np.inf
best_order = None
best_res = None

p_range = range(0, 4)
d_range = range(0, 2)
q_range = range(0, 4)

for p in p_range:
    for d in d_range:
        for q in q_range:

            model = SARIMAX(train,
                            order=(p, d, q),
                            trend='n',
                            enforce_stationarity=True,
                            enforce_invertibility=True)
            res_tmp = model.fit(disp=False)
            if res_tmp.aic < best_aic:
                best_aic = res_tmp.aic
                best_order = (p, d, q)
                best_res = res_tmp

print("\nBest model:", best_order)
p, d, q = best_order

# find estimerede AR og MA koefficienter

phi = [best_res.params.get(f"ar.L{i}", 0) for i in range(1, p+1)]
theta = [best_res.params.get(f"ma.L{i}", 0) for i in range(1, q+1)]


ar_est = np.r_[1, -np.array(phi)]
ma_est = np.r_[1, np.array(theta)]

print("\nEstimated AR coefficients:", ar_est)
print("Estimated MA coefficients:", ma_est)

sigma2_hat = best_res.params.get("sigma2", np.nan)
print("\nEstimated sigma² (innovation variance):", sigma2_hat)


# ACF og PACF 

lags = 30

acf_true  = arma_acf(ar_true, ma_true, lags=lags)
acf_est   = arma_acf(ar_est,  ma_est,  lags=lags)

pacf_true = arma_pacf(ar_true, ma_true, lags=lags)
pacf_est  = arma_pacf(ar_est,  ma_est,  lags=lags)

x_acf  = np.arange(len(acf_true))
x_pacf = np.arange(len(pacf_true))

# plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# acf
ax = axes[0]

# TRUE
ax.stem(x_acf, acf_true, linefmt='C0-', markerfmt='C0o', basefmt='k-')

# ESTIMATED
ax.stem(x_acf, acf_est,  linefmt='C1-', markerfmt='C1o', basefmt='k-')

ax.set_xlim(-0.15, 30)
ax.set_ylim(-1, 1)
ax.set_xlabel("Lag")
ax.set_ylabel("ACS")
ax.grid(True)
ax.set_title("")

# pacf
ax = axes[1]

# TRUE
ax.stem(x_pacf, pacf_true, linefmt='C0-', markerfmt='C0o', basefmt='k-')

# ESTIMATED
ax.stem(x_pacf, pacf_est,  linefmt='C1-', markerfmt='C1o', basefmt='k-')

ax.set_xlim(-0.15, 30)
ax.set_ylim(-1, 1)
ax.set_xlabel("Lag")
ax.set_ylabel("PACS")
ax.grid(True)
ax.set_title("")


legend_elements = [
    Line2D([0], [0], marker='o', color='C0', label='True ARIMA', markersize=6),
    Line2D([0], [0], marker='o', color='C1', label='Estimated ARIMA', markersize=6),]

axes[0].legend(handles=legend_elements, frameon=False)
axes[1].legend(handles=legend_elements, frameon=False)

plt.tight_layout()
plt.savefig("acf_pacf_true_vs_estimated.pdf")
plt.close()
