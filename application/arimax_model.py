import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

csv_path = 'pv_production_june1.csv'
target_col = 'pv_production'
latitude, longitude = 48.6727, 12.6931
period = 96

df = pd.read_csv(csv_path, parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
y = df[target_col].astype(float)

def fetch_open_meteo_data(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()

pv_times = df.index.strftime('%Y-%m-%dT%H:%M')
start_date = pv_times[0][:10]
end_date = pv_times[-1][:10]
weather_url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}"
    f"&hourly=temperature_2m,cloudcover_mid,direct_radiation&timezone=Europe/Berlin"
)
weather_data = fetch_open_meteo_data(weather_url)

if weather_data:
    hourly = weather_data.get('hourly', {})
    weather_times = hourly.get('time', [])
    #temp = hourly.get('temperature_2m', [np.nan]*len(weather_times))
    cloud = hourly.get('cloudcover_mid', [np.nan]*len(weather_times))
    direct = hourly.get('direct_radiation', [np.nan]*len(weather_times))

    weather_df = pd.DataFrame({
        'time': pd.to_datetime(weather_times),
        #'temperature': temp,
        'cloudcover_mid': cloud,
        'direct_radiation': direct,
    }).set_index('time')

    weather_on_pv = weather_df.reindex(df.index, method='nearest')
else:
    weather_on_pv = pd.DataFrame(index=df.index)

X = weather_on_pv.copy()

# --- Train/Validation split ---
val_h = int(period * 1.5)  # 1.5 * period (96) = 144 => 36 timer (test set)

X_train, X_val = X.iloc[:-val_h, :], X.iloc[-val_h:, :]
y_train, y_val = y.iloc[:-val_h], y.iloc[-val_h:]

# --- Resample til hourly ---
y_hour = y.resample('H').mean()
X_hour = X.resample('H').mean()

val_h_hours = int(val_h / 4)
y_train_hour, y_val_hour = y_hour.iloc[:-val_h_hours], y_hour.iloc[-val_h_hours:]
X_train_hour = X_hour.iloc[:-val_h_hours, :]
X_val_hour = X_hour.iloc[-val_h_hours:, :]

# --- Kun timemodeller (ingen global fallback) ---
hour_models = {}
hour_results = {}
min_samples_per_hour = 14
p, d, q = 2, 1, 0

for h in range(24):
    idx_train_h = y_train_hour.index[y_train_hour.index.hour == h]
    y_train_h = y_train_hour.loc[idx_train_h].dropna()
    X_train_h = X_train_hour.loc[idx_train_h].reindex(y_train_h.index).fillna(method='ffill')

    if len(y_train_h) < min_samples_per_hour:
        hour_models[h] = None
        hour_results[h] = None
        continue

    try:
        model_h = SARIMAX(
            endog=y_train_h,
            exog=X_train_h,
            order=(p, d, q),
            enforce_stationarity=True,
            enforce_invertibility=True,
            trend='n'
        )
        res_h = model_h.fit(disp=False, maxiter=100)
        hour_models[h] = model_h
        hour_results[h] = res_h
    except:
        hour_models[h] = None
        hour_results[h] = None

# --- Automatic model selection per hour (try multiple (p,d,q) orders)
best_orders = {}
best_results = {}

candidate_orders = [(1,1,0), (2,1,0), (3,1,0), (1,1,1), (2,1,1), (1,0,0), (0,0,0), (0,0,3), (2,0,0), (2,0,1)]

for h in range(24):
    idx_train_h = y_train_hour.index[y_train_hour.index.hour == h]
    y_train_h = y_train_hour.loc[idx_train_h].dropna()
    if len(y_train_h) < min_samples_per_hour:
        best_orders[h] = None
        best_results[h] = None
        continue

    X_train_h = X_train_hour.loc[idx_train_h].reindex(y_train_h.index).ffill()

    best_aic = np.inf
    best_model = None
    best_res = None
    best_order = None

    for order in candidate_orders:
        try:
            model = SARIMAX(endog=y_train_h, exog=X_train_h, order=order, trend='n', enforce_stationarity=True, enforce_invertibility=True)
            res = model.fit(disp=False)
            if res.aic < best_aic:
                best_aic = res.aic
                best_model = model
                best_res = res
                best_order = order
        except:
            continue

    best_orders[h] = best_order
    best_results[h] = best_res

hour_results = best_results

# --- Evaluate models (AIC, BIC, Ljung-Box) per hour
model_diagnostics = {}
from statsmodels.stats.diagnostic import acorr_ljungbox

for h in range(24):
    res = hour_results.get(h)
    if res is None:
        model_diagnostics[h] = None
        continue

    aic = res.aic
    bic = res.bic
    ljung = acorr_ljungbox(res.resid, lags=[10], return_df=True)

    model_diagnostics[h] = {
        'AIC': aic,
        'BIC': bic,
        'Order': best_orders[h],
        'LjungBox_p': ljung['lb_pvalue'].iloc[0]
    }

print("--- Model Diagnostics per hour (auto-selected) ---")
for h in range(24):
    diag = model_diagnostics[h]
    if diag is None:
        print(f"Hour {h}: no model (insufficient data)")
    else:
        print(f"Hour {h}: Order={diag['Order']}, AIC={diag['AIC']:.2f}, BIC={diag['BIC']:.2f}, LjungBox_p={diag['LjungBox_p']:.4f}")

# --- Generate residual plots per hour
import matplotlib.pyplot as plt
import os
os.makedirs('residual_plots', exist_ok=True)

for h in range(24):
    res = hour_results.get(h)
    if res is None:
        continue

    plt.figure(figsize=(8,3))
    plt.plot(res.resid)
    plt.title(f"Residuals Hour {h}")
    plt.tight_layout()
    plt.savefig(f"residual_plots/hour_{h}_residuals.pdf")
    plt.close()

print("Residual plots saved to folder 'residual_plots/'.")

# --- Plot ACF & PACF per hour
os.makedirs('acf_pacf_plots', exist_ok=True)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

for h in range(24):
    idx_train_h = y_train_hour.index[y_train_hour.index.hour == h]
    y_train_h = y_train_hour.loc[idx_train_h].dropna()
    if len(y_train_h) < min_samples_per_hour:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(10,3))
    plot_acf(y_train_h, ax=axes[0], alpha=None)
    axes[0].set_title(f"ACF Hour {h}")

    plot_pacf(y_train_h, ax=axes[1], method='ywm', alpha=None)
    axes[1].set_title(f"PACF Hour {h}")

    plt.tight_layout()
    plt.savefig(f"acf_pacf_plots/hour_{h}_acf_pacf.pdf")
    plt.close()

print("ACF/PACF plots saved to folder 'acf_pacf_plots/'.")
import matplotlib.pyplot as plt
import os
os.makedirs('residual_plots', exist_ok=True)

for h in range(24):
    res = hour_results.get(h)
    if res is None:
        continue

    plt.figure(figsize=(8,3))
    plt.plot(res.resid)
    plt.title(f"Residuals Hour {h}")
    plt.tight_layout()
    plt.savefig(f"residual_plots/hour_{h}_residuals.pdf")
    plt.close()

print("Residual plots saved to folder 'residual_plots/'.")

# --- Forecast hourly (AIC, BIC, Ljung-Box) per hour
model_diagnostics = {}
from statsmodels.stats.diagnostic import acorr_ljungbox

for h in range(24):
    res = hour_results.get(h)
    if res is None:
        model_diagnostics[h] = None
        continue

    # AIC / BIC
    aic = res.aic
    bic = res.bic

    # Ljung-Box test on residuals
    ljung = acorr_ljungbox(res.resid, lags=[10], return_df=True)

    model_diagnostics[h] = {
        'AIC': aic,
        'BIC': bic,
        'LjungBox_p': ljung['lb_pvalue'].iloc[0]
    }

print("--- Model Diagnostics per hour ---")
for h in range(24):
    diag = model_diagnostics[h]
    if diag is None:
        print(f"Hour {h}: no model (insufficient data)")
    else:
        print(f"Hour {h}: AIC={diag['AIC']:.2f}, BIC={diag['BIC']:.2f}, LjungBox_p={diag['LjungBox_p']:.4f}")

# --- Forecast hourly ---
preds_hour = []
preds_hour_lower = []
preds_hour_upper = []
idxs = []

for t_idx in y_val_hour.index:
    h = t_idx.hour
    res_h = hour_results.get(h)

    if res_h is None:
        preds_hour.append(np.nan)
        preds_hour_lower.append(np.nan)
        preds_hour_upper.append(np.nan)
        idxs.append(t_idx)
        continue

    exog_row = X_val_hour.loc[[t_idx]].reindex([t_idx]).fillna(method='ffill')

    pred_res = res_h.get_forecast(steps=1, exog=exog_row)
    mean_val = float(pred_res.predicted_mean.iloc[0])
    ci = pred_res.conf_int(alpha=0.05).iloc[0]

    preds_hour.append(mean_val)
    preds_hour_lower.append(float(ci.iloc[0]))
    preds_hour_upper.append(float(ci.iloc[1]))
    idxs.append(t_idx)

preds_hour_series = pd.Series(preds_hour, index=idxs)
preds_hour_lower_series = pd.Series(preds_hour_lower, index=idxs)
preds_hour_upper_series = pd.Series(preds_hour_upper, index=idxs)

# --- Map til 15-min ---
preds_15min_from_hour = preds_hour_series.reindex(y_val.index, method='ffill')
preds_15min_lower = preds_hour_lower_series.reindex(y_val.index, method='ffill')
preds_15min_upper = preds_hour_upper_series.reindex(y_val.index, method='ffill')

# --- Brug kun de sidste 24 punkter (24 × 15-min = 6 timer) ---
last_n = 96

y_val_last = y_val.tail(last_n)
preds_last = preds_15min_from_hour.tail(last_n)

rmse_hour_mapped = math.sqrt(mean_squared_error(y_val_last, preds_last))
mae_hour_mapped = mean_absolute_error(y_val_last, preds_last)

print(f"RMSE: {rmse_hour_mapped:.3f}")
print(f"MAE: {mae_hour_mapped:.3f}")



# ============================================================
# MODEL DIAGNOSTICS PER HOUR
# Residual time plot, QQ plot, histogram, and prediction-error plot
# ============================================================

os.makedirs('arimax_model', exist_ok=True)
os.makedirs('arimax_model/residual_timeseries', exist_ok=True)
os.makedirs('arimax_model/qq_plots', exist_ok=True)
os.makedirs('arimax_model/histograms', exist_ok=True)
os.makedirs('arimax_model/error_plots', exist_ok=True)


# --- Plot ---
# Vis kun sidste 24 timer (96 punkter) fra testsettet (testset er 36 timer i alt)
plot_n = 24 * 4# 24 timer * 4 (15-min) = 96 punkter
y_plot = y_val.tail(plot_n)
preds_plot = preds_15min_from_hour.tail(plot_n)
preds_plot_lower = preds_15min_lower.tail(plot_n)
preds_plot_upper = preds_15min_upper.tail(plot_n)

plt.figure(figsize=(12,5))
plt.plot(y_plot.index, y_plot, label='Test set')
plt.plot(preds_plot.index, preds_plot, label='Prediction')
plt.fill_between(preds_plot.index, preds_plot_lower, preds_plot_upper, alpha=0.2, label='95% CI')
plt.legend()
plt.tight_layout()
plt.savefig(f"arimax_model/plot.pdf")
plt.close()



from statsmodels.graphics.gofplots import qqplot

for h in range(24):
    res = hour_results.get(h)
    if res is None:
        continue

    resid = res.resid.dropna()

    # 1. Residual time series ---------------------------------
    plt.figure(figsize=(8, 3))
    # små prikker for hvert punkt + linje
    plt.plot(resid.index, resid.values, marker='o', linestyle='-', linewidth=1)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f"Residual Time Series - Hour {h}")
    plt.tight_layout()
    plt.savefig(f"arimax_model/residual_timeseries/hour_{h}_residuals.pdf")
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
    plt.title(f"QQ Plot - Hour {h}")
    plt.tight_layout()
    plt.savefig(f"arimax_model/qq_plots/hour_{h}_qq.pdf")
    plt.close()

    # 3. Histogram of residuals --------------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=20, alpha=0.7)
    plt.axvline(np.mean(resid), color='red', linestyle='--', label='Mean')
    plt.title(f"Residual Histogram - Hour {h}")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"arimax_model/histograms/hour_{h}_hist.pdf")
    plt.close()

    # 4. Prediction-error plot: residuals vs. fitted values -----
    fitted_vals = res.fittedvalues.dropna()

    # align index with residuals
    common_idx = resid.index.intersection(fitted_vals.index)
    resid_aligned = resid.loc[common_idx]
    fitted_aligned = fitted_vals.loc[common_idx]

    plt.figure(figsize=(6, 4))
    # allerede scatter — sørg for små prikker (s angiver punktstørrelse)
    plt.scatter(fitted_aligned, resid_aligned, s=10, alpha=0.7)
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"Prediction Error Plot - Hour {h}")
    plt.xlabel("Fitted values (1-step ahead)")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(f"arimax_model/error_plots/hour_{h}_errorplot.pdf")
    plt.close()

print("Full diagnostic plots saved in 'arimax_model/'")
