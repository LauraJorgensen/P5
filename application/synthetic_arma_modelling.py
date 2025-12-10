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
y = pd.Series(arma.generate_sample(nsample=30, burnin=200))

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

# udregn teoretisk ACF og PACF for både true og estimeret ARMA

lags = 30

acf_true  = arma_acf(ar_true, ma_true, lags=lags)
acf_est   = arma_acf(ar_est,  ma_est,  lags=lags)

pacf_true = arma_pacf(ar_true, ma_true, lags=lags)
pacf_est  = arma_pacf(ar_est,  ma_est,  lags=lags)

x_acf_true  = np.arange(len(acf_true))   # korrekt længde
x_acf_est   = np.arange(len(acf_est))
x_pacf_true = np.arange(len(pacf_true))  # korrekt længde
x_pacf_est  = np.arange(len(pacf_est))

# Plot ACF og PACF: True vs Estimated

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

axes[0].stem(x_acf_true, acf_true, linefmt='b-', markerfmt='bo', basefmt='k',
             label="True ARMA")
axes[0].stem(x_acf_est, acf_est, linefmt='r-', markerfmt='ro', basefmt='k',
             label="Estimated ARMA")
axes[0].set_xlabel("Lag")
axes[0].set_ylabel("ACS")
axes[0].legend()

axes[1].stem(x_pacf_true, pacf_true, linefmt='b-', markerfmt='bo', basefmt='k',
             label="True ARMA")
axes[1].stem(x_pacf_est, pacf_est, linefmt='r-', markerfmt='ro', basefmt='k',
             label="Estimated ARMA")
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("PACS")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"synthetic_arma_acf_pacf.pdf")
plt.close()

