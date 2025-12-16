import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# --- Konfiguration ---
os.makedirs('arimax_model', exist_ok=True)
os.makedirs('arimax_model/residual_timeseries', exist_ok=True)
os.makedirs('arimax_model/qq_plots', exist_ok=True)

csv_path = 'data/pv_production_june_clean.csv'
target_col = 'pv_production'
latitude, longitude = 48.6727, 12.6931
period = 96 # antal kvarter på et døgn

df = pd.read_csv(csv_path, parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
y = df[target_col].astype(float)


# --- Import og align exogenous variables ---
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
    f"&hourly=temperature_2m,cloudcover_mid,direct_radiation&timezone=UTC"
)
weather_data = fetch_open_meteo_data(weather_url)

if weather_data:
    hourly = weather_data.get('hourly', {})
    weather_times = hourly.get('time', [])
    cloud = hourly.get('cloudcover_mid', [np.nan]*len(weather_times))
    direct = hourly.get('direct_radiation', [np.nan]*len(weather_times))
    weather_df = pd.DataFrame({'time': pd.to_datetime(weather_times),'cloudcover_mid': cloud,'direct_radiation': direct,}).set_index('time')
    weather_on_pv = weather_df.reindex(df.index, method='nearest')
else:
    weather_on_pv = pd.DataFrame(index=df.index)

X = weather_on_pv.copy()

# --- Inddel test og training set på 15-min data ---
val_h = int(period * 1.5)  # 36 timer

X_train, X_val = X.iloc[:-val_h, :], X.iloc[-val_h:, :]
y_train, y_val = y.iloc[:-val_h], y.iloc[-val_h:]

# --- Resample til hourly ---
y_hour = y.resample('H').mean()
X_hour = X.resample('H').mean()

# --- Inddel test og training set på hourly data ---
val_h_hours = int(val_h / 4)
y_train_hour, y_val_hour = y_hour.iloc[:-val_h_hours], y_hour.iloc[-val_h_hours:]
X_train_hour = X_hour.iloc[:-val_h_hours, :]
X_val_hour = X_hour.iloc[-val_h_hours:, :]

# --- Automatisk modelvalg pr. time ---
candidate_orders = [(p, d, q)
    for p in range(0, 4)
    for d in range(0, 2)
    for q in range(0, 4)]

best_orders = {}
best_results = {}

for h in range(24):
    idx_train_h = y_train_hour.index[y_train_hour.index.hour == h]
    y_train_h = y_train_hour.loc[idx_train_h].dropna()
    X_train_h = X_train_hour.loc[idx_train_h].reindex(y_train_h.index).ffill()
    best_aic = np.inf
    best_model = None
    best_res = None
    best_order = None

    for order in candidate_orders:
        model = SARIMAX(endog=y_train_h, exog=X_train_h, order=order, trend='n', enforce_stationarity=True, enforce_invertibility=True)
        res = model.fit(disp=False)
        if res.aic < best_aic:
            best_aic = res.aic
            best_model = model
            best_res = res
            best_order = order
    best_orders[h] = best_order
    best_results[h] = best_res

hour_results = best_results

# --- Diagnostics pr. time ---
model_diagnostics = {}

for h in range(24):
    res = hour_results.get(h)
    aic = res.aic
    bic = res.bic
    ljung = acorr_ljungbox(res.resid, lags=[10], return_df=True)
    model_diagnostics[h] = {'AIC': aic,'BIC': bic,'Order': best_orders[h],'LjungBox_p': ljung['lb_pvalue'].iloc[0]}

print("--- Selected Hourly Models ---")
for h in range(24):
    diag = model_diagnostics[h]
    print(f"Hour {h}: Order={diag['Order']}, AIC={diag['AIC']:.2f}, LjungBox_p={diag['LjungBox_p']:.4f}")

# --- Forecast ---
preds_hour = []
preds_hour_lower = []
preds_hour_upper = []
idxs = []

for t_idx in y_val_hour.index:
    h = t_idx.hour
    res_h = hour_results.get(h)

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

# --- Model evaluation metrics for last 24 timer ---
last_n = 96
y_val_last = y_val.tail(last_n)
preds_last = preds_15min_from_hour.tail(last_n)

rmse_hour_mapped = math.sqrt(mean_squared_error(y_val_last, preds_last))
mae_hour_mapped = mean_absolute_error(y_val_last, preds_last)

print(f"RMSE: {rmse_hour_mapped:.3f}")
print(f"MAE: {mae_hour_mapped:.3f}")

# --- Plot forecast vs obs for last 24 timer ---
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
plt.ylabel('Normalized PV Production')
plt.xlabel("Time")
plt.grid(True)
plt.ylim(-0.6,1.2)
plt.xlim(y_plot.index.min(), y_plot.index.max())
plt.tight_layout()
plt.savefig(f"arimax_model/arimaxplot.pdf")
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
    plt.savefig(f"arimax_model/residual_timeseries/hour_{h}_residuals.pdf")
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
    plt.savefig(f"arimax_model/qq_plots/hour_{h}_qq.pdf")
    plt.close()

 
