import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Konfiguration ---
csv_path = 'data/pv_production_june_clean.csv'
target_col = 'pv_production'

period_15min = 96              
test_hours = 36                
test_15min = test_hours * 4    

df = pd.read_csv(csv_path, parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
y = df[target_col].astype(float)

# --- Inddel test og training set på 15-min data ---
y_train = y.iloc[:-test_15min]
y_test = y.iloc[-test_15min:]         

# --- Resample til hourly ---
y_hour = y.resample('H').mean()

# --- Inddel test og training set på hourly data ---
test_hours = int(test_15min / 4) # antal timer i test
y_train_hour = y_hour.iloc[:-test_hours]
y_test_hour = y_hour.iloc[-test_hours:]

# --- Forecasts using the Persistence Model --- 
preds_hour = []
preds_hour_lower = []
preds_hour_upper = []
idxs = []

for t_idx in y_test_hour.index:
    h = t_idx.hour
    idx_train_h = y_train_hour.index[y_train_hour.index.hour == h]
    y_train_h = y_train_hour.loc[idx_train_h].dropna()

    # sidste træningsværdi og tid
    last_val = y_train_h.iloc[-1]
    last_time = y_train_h.index[-1]
    
    diffs = y_train_h.diff().dropna()
    sigma = diffs.std()
    if pd.isna(sigma) or sigma == 0:
        sigma = 1e-6  # fallback

    # Beregn antal skridt i forecasts
    if pd.isna(last_time) or t_idx <= last_time:
        k = 1
    else:
        days = (t_idx - last_time) / pd.Timedelta(days=1)
        k = max(1, int(round(days)))

    # 95% CI for k-step ahead
    margin = 1.96 * sigma * np.sqrt(k)
    ci_lower = last_val - margin
    ci_upper = last_val + margin

    preds_hour.append(float(last_val))
    preds_hour_lower.append(float(ci_lower))
    preds_hour_upper.append(float(ci_upper))
    idxs.append(t_idx)

preds_hour_series = pd.Series(preds_hour, index=idxs)
preds_hour_lower_series = pd.Series(preds_hour_lower, index=idxs)
preds_hour_upper_series = pd.Series(preds_hour_upper, index=idxs)

# --- Map til 15-min ---
preds_15min = preds_hour_series.reindex(y_test.index, method='ffill')
preds_15min_lower = preds_hour_lower_series.reindex(y_test.index, method='ffill')
preds_15min_upper = preds_hour_upper_series.reindex(y_test.index, method='ffill')

# --- Model evaluation metrics for last 24 timer ---
y_test_last = y_test.tail(period_15min)
preds_last = preds_15min.tail(period_15min)

rmse = math.sqrt(mean_squared_error(y_test_last, preds_last))
mae = mean_absolute_error(y_test_last, preds_last)

print(f"\nRMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")

# --- Plot forecast vs obs for last 24 timer ---
last_day_test = y_test.iloc[-period_15min:]

plt.figure(figsize=(10, 5))
plt.plot(last_day_test.index, last_day_test, label='Test set')
plt.plot(last_day_test.index, preds_15min.loc[last_day_test.index], label='Prediction')
plt.ylabel('Normalized PV Production')
plt.xlabel("Time")
plt.grid(True)
plt.ylim(-0.6,1.2)
plt.xlim(last_day_test.index.min(), last_day_test.index.max())
plt.tight_layout()
plt.fill_between(last_day_test.index,preds_15min_lower.loc[last_day_test.index],preds_15min_upper.loc[last_day_test.index],alpha=0.25,label='95% CI')
plt.legend()
plt.tight_layout()
plt.savefig(f"Persistence_Model.pdf")
plt.close()
