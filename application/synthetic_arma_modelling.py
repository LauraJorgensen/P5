import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.gofplots import qqplot

# =======================================================
# 1. Generate synthetic ARMA data
# =======================================================

np.random.seed(42)

# Define ARMA(2,1) with known parameters
# AR part:   y_t = 0.7 y_{t-1} - 0.2 y_{t-2} + e_t
# MA part:   e_t + 0.5 e_{t-1}

ar = np.array([1, -0.7, 0.2])   # Note sign convention
ma = np.array([1, 0.9])

arma = ArmaProcess(ar, ma)
y = arma.generate_sample(nsample=500)

y = pd.Series(y)

# =======================================================
# 2. Fit ARMA/ARIMA model to the data (grid search by AIC)
# =======================================================

import warnings
warnings.filterwarnings("ignore")

# AIC-based grid-search: enforce invertibility, cap q
best_aic = np.inf
best_order_aic = None
best_res_aic = None

p_range = range(0, 4)     # be konservativ med p
d_range = range(0, 2)
q_range = range(0, 4)     # cap q til 3 for at undgå MA-overfitting

print("\nStarting AIC grid search (enforce_invertibility=True)...")
for p in p_range:
    for d in d_range:
        for q in q_range:
            order = (p, d, q)
            try:
                mod = SARIMAX(y, order=order, trend='n',
                              enforce_stationarity=True,
                              enforce_invertibility=True)
                res_tmp = mod.fit(disp=False, maxiter=100)
            except Exception:
                continue

            aic = res_tmp.aic
            print(f" Tested order {order} -> AIC {aic:.3f}")
            if aic < best_aic:
                best_aic = aic
                best_order_aic = order
                best_res_aic = res_tmp

# vælg AIC-model
best_order = best_order_aic
best_res = best_res_aic

print("\nSelected (AIC):", best_order, "AIC:", best_aic)

print("\n=== TRUE PARAMETERS ===")
print("AR(1) = 0.7")
print("AR(2) = -0.2")
print("MA(1) = 0.9")

# Print estimated parameters (if model fitted)
if best_res is not None:
    print("\n=== ESTIMATED PARAMETERS ===")
    try:
        params = best_res.params
        bse = getattr(best_res, "bse", None)
        for name in params.index:
            est = params[name]
            se = bse.get(name, np.nan) if bse is not None else np.nan
            print(f"{name}: {est:.4f} (se={se:.4f})")
    except Exception:
        print(best_res.params)
else:
    print("\nNo fitted model found.")

# =======================================================
# 3. Extract residuals and fitted values
# =======================================================

residuals = best_res.resid.dropna()
fitted = best_res.fittedvalues.dropna()

# Align (to avoid index mismatch)
common = residuals.index.intersection(fitted.index)
residuals = residuals.loc[common]
fitted = fitted.loc[common]

# =======================================================
# 4. Make output folder
# =======================================================

os.makedirs("synthetic_diagnostics", exist_ok=True)

# =======================================================
# 5. Plot: Residual time series
# =======================================================

plt.figure(figsize=(10,3))
plt.plot(residuals.index, residuals.values, marker='o', markersize=3, linewidth=1)
plt.axhline(0, color='black', linewidth=1)
plt.title("Residual Time Series")
plt.tight_layout()
plt.savefig("synthetic_diagnostics/residual_timeseries.png")
plt.close()

# =======================================================
# 6. Plot: Histogram of residuals
# =======================================================

plt.figure(figsize=(6,4))
plt.hist(residuals, bins=20, alpha=0.7)
plt.axvline(np.mean(residuals), color='red', linestyle='--', label='Mean')
plt.title("Residual Histogram")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("synthetic_diagnostics/residual_histogram.png")
plt.close()

# =======================================================
# 7. Plot: QQ plot
# =======================================================

plt.figure(figsize=(5,5))
qqplot(residuals, line='s', ax=plt.gca())
plt.title("QQ Plot of Residuals")
plt.tight_layout()
plt.savefig("synthetic_diagnostics/qq_plot.png")
plt.close()

# =======================================================
# 8. Plot: Prediction error plot (residual vs fitted)
# =======================================================

plt.figure(figsize=(6,4))
plt.scatter(fitted, residuals, s=10, alpha=0.7)
plt.axhline(0, color='black', linewidth=1)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Prediction Error Plot")
plt.tight_layout()
plt.savefig("synthetic_diagnostics/prediction_error_plot.png")
plt.close()

print("\nAll diagnostic plots saved in folder: synthetic_diagnostics/\n")
