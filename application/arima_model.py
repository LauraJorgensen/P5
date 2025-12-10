import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")


# --- Konfiguration ---
csv_path = 'pv_production_june1.csv'
target_col = 'pv_production'
period = 96  # 15-min pr døgn

# --- Data indlæsning ---
df = pd.read_csv(csv_path, parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
y = df[target_col].astype(float)

# --- Train/Validation split på 15-min data ---
# val_h = period  # antal 15-min punkter til validering
val_h = int(period * 1.5)  # 1.5 * period (96) = 144 => 36 timer (test set)
y_train, y_val = y.iloc[:-val_h], y.iloc[-val_h:]

# --- Resample til hourly (uden exogene) ---
y_hour = y.resample('h').mean()

# --- Opret hourly train/validation split (vigtigt) ---
# val_h er antal 15-min punkter i validation (f.eks. 96 = 24h)
val_h_hours = int(val_h / 4)  # 4 * 15min = 1 time
y_train_hour = y_hour.iloc[:-val_h_hours]
y_val_hour = y_hour.iloc[-val_h_hours:]

# --- Plot hvert hours time series (24 plots) ---
os.makedirs('hourly_series_plots', exist_ok=True)
for h in range(24):
    idx_h = y_hour.index[y_hour.index.hour == h]
    series_h = y_hour.loc[idx_h].dropna()
    if series_h.empty:
        continue
    plt.figure(figsize=(10, 3))
    plt.scatter(series_h.index, series_h.values, s=10)
    plt.title(f"Hourly Process - Hour {h}")
    plt.xlabel('Time')
    plt.ylabel('PV Production')
    plt.tight_layout()
    plt.savefig(f"hourly_series_plots/hour_{h}_series.pdf")
    plt.close()
print("Hourly processes plots saved to folder 'hourly_series_plots/'.")

# --- Automatisk modelvalg pr. time (KUN endogen, ingen exog) ---
min_samples_per_hour = 14
candidate_orders = [(p, d, q)
    for p in range(0, 4)
    for d in range(0, 2)
    for q in range(0, 4)]

best_orders = {}
hour_results = {}

for h in range(24):
    idx_train_h = y_train_hour.index[y_train_hour.index.hour == h]
    y_train_h = y_train_hour.loc[idx_train_h].dropna()

    if len(y_train_h) < min_samples_per_hour:
        best_orders[h] = None
        hour_results[h] = None
        continue

    best_aic = np.inf
    best_res = None
    best_order = None

    for order in candidate_orders:
        try:
            model = SARIMAX(
                endog=y_train_h,
                order=order,
                trend='n',
                enforce_stationarity=True,
                enforce_invertibility=True
            )
            res = model.fit(disp=False)
            if res.aic < best_aic:
                best_aic = res.aic
                best_res = res
                best_order = order
        except Exception:
            continue

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

    arima_model[h] = {
        'AIC': aic,
        'BIC': bic,
        'Order': best_orders[h],
        'LjungBox_p': ljung['lb_pvalue'].iloc[0]
    }

print("\n--- Selected Hourly Models ---")
for h in range(24):
    diag = arima_model[h]
    if diag is None:
        print(f"Hour {h}: no model (insufficient data)")
    else:
        print(
            f"Hour {h}: Order={diag['Order']}, AIC={diag['AIC']:.2f}, LjungBox_p={diag['LjungBox_p']:.4f}"
        )

# --- ACF & PACF pr. time (på træningsserien) ---
os.makedirs('acf_pacf_plots', exist_ok=True)

for h in range(24):
    idx_train_h = y_train_hour.index[y_train_hour.index.hour == h]
    y_train_h = y_train_hour.loc[idx_train_h].dropna()
    if len(y_train_h) < min_samples_per_hour:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    plot_acf(y_train_h, ax=axes[0], alpha=None)
    axes[0].set_title(f"ACS Hour {h}")
    plot_pacf(y_train_h, ax=axes[1], method='ywm', alpha=None)
    axes[1].set_title(f"PACS Hour {h}")
    plt.tight_layout()
    plt.savefig(f"acf_pacf_plots/hour_{h}_acf_pacf.pdf")
    plt.close()

print("ACF/PACF plots saved to folder 'acf_pacf_plots/'.")

# --- QQ plots per hour
os.makedirs('qq_plots', exist_ok=True)
from statsmodels.graphics.gofplots import qqplot

for h in range(24):
    res = hour_results.get(h)
    if res is None:
        continue

    fig = plt.figure(figsize=(4,4))
    qqplot(res.resid, line='s', ax=plt.gca())
    plt.title(f"QQ Plot Residuals Hour {h}")
    plt.tight_layout()
    plt.savefig(f"qq_plots/hour_{h}_qqplot.pdf")
    plt.close()

print("QQ plots saved to folder 'qq_plots/'.")
os.makedirs('acf_pacf_plots', exist_ok=True)

# --- Forecast (uden exogene) på hourly validation (multi-step per hour, CI vokser med horizon) ---
preds_hour = []
preds_hour_lower = []
preds_hour_upper = []
idxs = []

for h in range(24):
    res_h = hour_results.get(h)
    # timestamps i hourly validation for denne time (i rækkefølge)
    future_idx = y_val_hour.index[y_val_hour.index.hour == h]

    if res_h is None or len(future_idx) == 0:
        for t_idx in future_idx:
            preds_hour.append(np.nan)
            preds_hour_lower.append(np.nan)
            preds_hour_upper.append(np.nan)
            idxs.append(t_idx)
        continue

    # Multi-step forecast fra træningsslut: antal skridt = antal forekomster i valideringen for denne time
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

# Byg serier og map til 15-min
preds_hour_series = pd.Series(preds_hour, index=idxs).sort_index()
preds_hour_lower_series = pd.Series(preds_hour_lower, index=idxs).sort_index()
preds_hour_upper_series = pd.Series(preds_hour_upper, index=idxs).sort_index()

preds_15min_from_hour = preds_hour_series.reindex(y_val.index, method='ffill')
preds_15min_lower = preds_hour_lower_series.reindex(y_val.index, method='ffill')
preds_15min_upper = preds_hour_upper_series.reindex(y_val.index, method='ffill')

# --- Brug kun den sidste 24 dag ---
last_n = 96
y_val_last = y_val.tail(last_n)
preds_last = preds_15min_from_hour.tail(last_n)

rmse_hour_mapped = math.sqrt(mean_squared_error(y_val_last, preds_last))
mae_hour_mapped = mean_absolute_error(y_val_last, preds_last)

print(f"\nRMSE: {rmse_hour_mapped:.3f}")
print(f"MAE: {mae_hour_mapped:.3f}")



# ============================================================
# MODEL DIAGNOSTICS PER HOUR
# Residual time plot, QQ plot, histogram, and prediction-error plot
# ============================================================

os.makedirs('arima_model', exist_ok=True)
os.makedirs('arima_model/residual_timeseries', exist_ok=True)
os.makedirs('arima_model/qq_plots', exist_ok=True)
os.makedirs('arima_model/histograms', exist_ok=True)
os.makedirs('arima_model/error_plots', exist_ok=True)

# --- Plot forecast vs obs ---
plot_n = 24 * 4  # vis kun sidste 36 timer (144 punkter)
y_plot = y_val.tail(plot_n)
preds_plot = preds_15min_from_hour.tail(plot_n)
preds_plot_lower = preds_15min_lower.tail(plot_n)
preds_plot_upper = preds_15min_upper.tail(plot_n)

plt.figure(figsize=(12, 5))
plt.plot(y_plot.index, y_plot, label='Test set')
plt.plot(preds_plot.index, preds_plot, label='Prediction')
plt.fill_between(preds_plot.index, preds_plot_lower, preds_plot_upper, alpha=0.2, label='95% CI')
plt.legend()
plt.tight_layout()
plt.savefig(f"arima_model/plot.pdf")
plt.close()



from statsmodels.graphics.gofplots import qqplot

for h in range(24):
    res = hour_results.get(h)
    if res is None:
        continue

    resid = res.resid.dropna()

    # 1. Residual time series ---------------------------------
    plt.figure(figsize=(8, 3))

    # scatter i stedet for line-plot
    plt.scatter(resid.index, resid.values, s=10)  # s = punktstørrelse
    plt.axhline(0, color='black', linewidth=0.8)

    plt.title(f"Residual Time Plot - Hour {h}")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(f"arima_model/residual_timeseries/hour_{h}_residuals.pdf")
    plt.close()

    # 2. QQ plot ----------------------------------------------
    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    qqplot(resid, line='s', ax=ax)
    # sæt små prikker for de punkter der er markeret i qqplot
    for line in ax.get_lines():
        mk = line.get_marker()
        if mk not in (None, 'None', ''):
            line.set_markersize(3)
    plt.title(f"Q-Q Plot - Hour {h}")
    plt.tight_layout()
    plt.savefig(f"arima_model/qq_plots/hour_{h}_qq.pdf")
    plt.close()

 

print("Full diagnostic plots saved in 'arima_model/'")

