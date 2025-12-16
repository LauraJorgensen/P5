import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
import warnings
warnings.filterwarnings("ignore")

# --- Konfiguration ---
os.makedirs('hourly_series_plots', exist_ok=True)
os.makedirs('acf_pacf_plots', exist_ok=True)
os.makedirs('arima_model', exist_ok=True)
os.makedirs('arima_model/residual_timeseries', exist_ok=True)
os.makedirs('arima_model/qq_plots', exist_ok=True)

csv_path = 'data/pv_production_june_clean.csv'
target_col = 'pv_production'
period = 96  # antal kvarter på et døgn

df = pd.read_csv(csv_path, parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
y = df[target_col].astype(float)

# --- Test/training set på 15-min data ---
val_h = int(period * 1.5)  # 36 timer 
y_train, y_val = y.iloc[:-val_h], y.iloc[-val_h:]

# --- Resample til hourly ---
y_hour = y.resample('h').mean()

# --- Test/training set på hourly data ---
val_h_hours = int(val_h / 4)  # 4 * 15min = 1 time
y_train_hour = y_hour.iloc[:-val_h_hours]
y_val_hour = y_hour.iloc[-val_h_hours:]

# --- Plot hver time  ---
for h in range(24):
    idx_h = y_hour.index[y_hour.index.hour == h]
    series_h = y_hour.loc[idx_h].dropna()
    if series_h.empty:
        continue
    plt.figure(figsize=(10, 3))
    plt.scatter(series_h.index, series_h.values, s=10)
    plt.title(f"Hour {h}")
    plt.xlim(series_h.index.min(), series_h.index.max())
    plt.xlabel('Time')
    plt.grid(True)
    plt.ylabel('Normalized PV Production')
    plt.tight_layout()
    plt.savefig(f"hourly_series_plots/hour_{h}_series.pdf")
    plt.close()

# --- Automatisk modelvalg pr. time ---
candidate_orders = [(p, d, q)
    for p in range(0, 4)
    for d in range(0, 2)
    for q in range(0, 4)]

best_orders = {}
hour_results = {}

for h in range(24):
    idx_train_h = y_train_hour.index[y_train_hour.index.hour == h]
    y_train_h = y_train_hour.loc[idx_train_h].dropna()

    best_aic = np.inf
    best_res = None
    best_order = None

    for order in candidate_orders:
        model = SARIMAX(endog=y_train_h,order=order,trend='n',enforce_stationarity=True,enforce_invertibility=True)
        res = model.fit(disp=False)
        if res.aic < best_aic:
            best_aic = res.aic
            best_res = res
            best_order = order

    best_orders[h] = best_order
    hour_results[h] = best_res

# --- Diagnostics (AIC, BIC, Ljung-Box) pr. time ---
arima_model = {}

for h in range(24):
    res = hour_results.get(h)
    if res is None:
        arima_model[h] = None
        continue
    aic = res.aic
    bic = res.bic
    ljung = acorr_ljungbox(res.resid, lags=[10], return_df=True)
    arima_model[h] = {'AIC': aic,'BIC': bic,'Order': best_orders[h],'LjungBox_p': ljung['lb_pvalue'].iloc[0]}

print("\n--- Selected Hourly Models ---")
for h in range(24):
    diag = arima_model[h]
    print(f"Hour {h}: Order={diag['Order']}, AIC={diag['AIC']:.2f}, LjungBox_p={diag['LjungBox_p']:.4f}")

# --- ACF & PACF pr. time ---
for h in range(24):
    idx_train_h = y_train_hour.index[y_train_hour.index.hour == h]
    y_train_h = y_train_hour.loc[idx_train_h].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    plot_acf(y_train_h, ax=axes[0], alpha=None)
    axes[0].grid(True)
    axes[0].set_title(f"Hour {h}")
    axes[0].set_ylabel("ACS")
    axes[0].set_xlabel("Lag")
    plot_pacf(y_train_h, ax=axes[1], method='ywm', alpha=None)
    axes[1].set_title(f"Hour {h}")
    axes[1].set_ylabel("PACS")
    axes[1].set_xlabel("Lag")
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(f"acf_pacf_plots/hour_{h}_acf_pacf.pdf")
    plt.close()


# --- Forecast  ---
preds_hour = []
preds_hour_lower = []
preds_hour_upper = []
idxs = []

for h in range(24):
    res_h = hour_results.get(h)
    future_idx = y_val_hour.index[y_val_hour.index.hour == h]

    # Forecast fra træningssættet. Antal steps afhænger af timen. 
    steps = len(future_idx)
    pred_res = res_h.get_forecast(steps=steps)
    means = pred_res.predicted_mean
    ci = pred_res.conf_int(alpha=0.05)

    # Tilknyt hvert forecast-skridt til tilsvarende timestamp
    for i, t_idx in enumerate(future_idx):
        preds_hour.append(float(means.iloc[i]))
        preds_hour_lower.append(float(ci.iloc[i, 0]))
        preds_hour_upper.append(float(ci.iloc[i, 1]))
        idxs.append(t_idx)

preds_hour_series = pd.Series(preds_hour, index=idxs).sort_index()
preds_hour_lower_series = pd.Series(preds_hour_lower, index=idxs).sort_index()
preds_hour_upper_series = pd.Series(preds_hour_upper, index=idxs).sort_index()

# --- Map til 15-min ---
preds_15min_from_hour = preds_hour_series.reindex(y_val.index, method='ffill')
preds_15min_lower = preds_hour_lower_series.reindex(y_val.index, method='ffill')
preds_15min_upper = preds_hour_upper_series.reindex(y_val.index, method='ffill')

# --- Model evaluation metrics for last 24 timer ---
last_n = 96
y_val_last = y_val.tail(last_n)
preds_last = preds_15min_from_hour.tail(last_n)

rmse_hour_mapped = math.sqrt(mean_squared_error(y_val_last, preds_last))
mae_hour_mapped = mean_absolute_error(y_val_last, preds_last)

print(f"\nRMSE: {rmse_hour_mapped:.3f}")
print(f"MAE: {mae_hour_mapped:.3f}")

# --- Plot forecast vs obs for last 24 timer ---
plot_n = 24 * 4  # vis kun sidste 36 timer (144 punkter)
y_plot = y_val.tail(plot_n)
preds_plot = preds_15min_from_hour.tail(plot_n)
preds_plot_lower = preds_15min_lower.tail(plot_n)
preds_plot_upper = preds_15min_upper.tail(plot_n)

plt.figure(figsize=(10, 5))
plt.plot(y_plot.index, y_plot, label='Test set')
plt.plot(preds_plot.index, preds_plot, label='Prediction')
plt.fill_between(preds_plot.index, preds_plot_lower, preds_plot_upper, alpha=0.2, label='95% CI')
plt.legend()
plt.ylabel('Normalized PV Production')
plt.xlabel("Time")
plt.grid(True)
plt.ylim(-0.6,1.2)
plt.xlim(y_plot.index.min(), y_plot.index.max())
plt.tight_layout()
plt.savefig(f"arima_model/arimaplot.pdf")
plt.close()

# --- Model diagnostic plots ---
for h in range(24):
    res = hour_results.get(h)
    resid = res.resid.dropna()

    # --- Residual time series ---
    plt.figure(figsize=(8, 3))
    plt.scatter(resid.index, resid.values, s=10)  # s = punktstørrelse
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f"Hour {h}")
    plt.ylabel("Residuals")
    plt.xlabel("Time")
    plt.xlim(resid.index.min(), resid.index.max())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"arima_model/residual_timeseries/hour_{h}_residuals.pdf")
    plt.close()

    # --- QQ plot ---
    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    qqplot(resid, line='s', ax=ax)
    for line in ax.get_lines():
        mk = line.get_marker()
        if mk not in (None, 'None', ''):
            line.set_markersize(3)
    plt.title(f"Hour {h}")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"arima_model/qq_plots/hour_{h}_qq.pdf")
    plt.close()

