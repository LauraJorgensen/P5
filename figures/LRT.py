
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

# --- Konfiguration ---
csv_path = 'pv_production_june1.csv'
target_col = 'pv_production'
latitude, longitude = 48.6727, 12.6931
period = 96

# --- Data indlæsning ---
df = pd.read_csv(csv_path, parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
y = df[target_col].astype(float)

# --- Hent vejrdata ---
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
    f"&hourly=temperature_2m,cloud_cover,direct_radiation,diffuse_radiation,cloudcover_low,cloudcover_mid,cloudcover_high&timezone=UTC"
)
weather_data = fetch_open_meteo_data(weather_url)

if weather_data:
    hourly = weather_data.get('hourly', {})
    weather_times = hourly.get('time', [])
    
    def _get(field):
        return hourly.get(field, [np.nan] * len(weather_times))
    
    weather_df = pd.DataFrame({
        'time': pd.to_datetime(weather_times),
        'temperature': _get('temperature_2m'),
        'cloud_cover': _get('cloud_cover'),
        'direct_radiation': _get('direct_radiation'),
        'diffuse_radiation': _get('diffuse_radiation'),
        'cloudcover_low': _get('cloudcover_low'),
        'cloudcover_mid': _get('cloudcover_mid'),
        'cloudcover_high': _get('cloudcover_high')
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

# =============================================================
# LIKELIHOOD RATIO TEST (LRT) FOR EXOGENOUS VARIABLES
# =============================================================
from scipy.stats import chi2

exog_variables_to_test = ['cloud_cover', 'temperature', 'direct_radiation', 
                          'diffuse_radiation', 'cloudcover_low', 'cloudcover_mid', 
                          'cloudcover_high']

# Filter to only include variables that exist in X
exog_variables_to_test = [v for v in exog_variables_to_test if v in X.columns]

print("\n" + "="*60)
print("LIKELIHOOD RATIO TEST FOR EXOGENOUS VARIABLES")
print("="*60)

p, d, q = 2, 1, 0
min_samples_per_hour = 14
lrt_results_by_var = {var: {} for var in exog_variables_to_test}

for var in exog_variables_to_test:
    print(f"\nTesting variable: {var}")
    print("-" * 40)
    
    for h in range(24):
        idx_train_h = y_train_hour.index[y_train_hour.index.hour == h]
        y_train_h = y_train_hour.loc[idx_train_h].dropna()
        
        if len(y_train_h) < min_samples_per_hour:
            lrt_results_by_var[var][h] = None
            continue
        
        X_train_h_single = X_train_hour.loc[idx_train_h, [var]].reindex(y_train_h.index).ffill()
        
        # Fit model WITHOUT exogenous variable (null model)
        try:
            model_null = SARIMAX(
                endog=y_train_h,
                order=(p, d, q),
                enforce_stationarity=True,
                enforce_invertibility=True,
                trend='n'
            )
            res_null = model_null.fit(disp=False, maxiter=100)
            ll_null = res_null.llf
        except:
            lrt_results_by_var[var][h] = None
            continue
        
        # Fit model WITH exogenous variable (alternative model)
        try:
            model_alt = SARIMAX(
                endog=y_train_h,
                exog=X_train_h_single,
                order=(p, d, q),
                enforce_stationarity=True,
                enforce_invertibility=True,
                trend='n'
            )
            res_alt = model_alt.fit(disp=False, maxiter=100)
            ll_alt = res_alt.llf
        except:
            lrt_results_by_var[var][h] = None
            continue
        
        # Likelihood ratio test statistic
        lr_stat = 2 * (ll_alt - ll_null)
        df_diff = X_train_h_single.shape[1]  # number of additional parameters
        p_value = 1 - chi2.cdf(lr_stat, df_diff)
        
        lrt_results_by_var[var][h] = {
            'll_null': ll_null,
            'll_alt': ll_alt,
            'lr_stat': lr_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Print summary for this variable
    significant_hours = [h for h in range(24) 
                        if lrt_results_by_var[var].get(h) is not None 
                        and lrt_results_by_var[var][h]['significant']]
    
    if significant_hours:
        print(f"  Significant at α=0.05 for hours: {significant_hours}")
    else:
        print(f"  Not significant for any hour")

# Print detailed LRT results
print("\n" + "="*60)
print("DETAILED LRT RESULTS")
print("="*60)

for var in exog_variables_to_test:
    print(f"\n{var}:")
    print(f"{'Hour':<6} {'LR Stat':<10} {'p-value':<10} {'Significant':<12}")
    print("-" * 40)
    for h in range(24):
        result = lrt_results_by_var[var].get(h)
        if result is None:
            print(f"{h:<6} {'N/A':<10} {'N/A':<10} {'N/A':<12}")
        else:
            sig_mark = "Yes" if result['significant'] else "No"
            print(f"{h:<6} {result['lr_stat']:<10.3f} {result['p_value']:<10.4f} {sig_mark:<12}")

# =============================================================
# CREATE HEATMAP OF LRT RESULTS
# =============================================================
import seaborn as sns

# Create matrices for heatmap
p_value_matrix = np.zeros((len(exog_variables_to_test), 24))
lr_stat_matrix = np.zeros((len(exog_variables_to_test), 24))

for i, var in enumerate(exog_variables_to_test):
    for h in range(24):
        result = lrt_results_by_var[var].get(h)
        if result is not None:
            p_value_matrix[i, h] = result['p_value']
            lr_stat_matrix[i, h] = result['lr_stat']
        else:
            p_value_matrix[i, h] = np.nan
            lr_stat_matrix[i, h] = np.nan

# Filter hours: remove 0,1,2,20,21,22,23 (keep hours 3-19)
hours_to_plot = list(range(3, 20))
p_value_matrix_filtered = p_value_matrix[:, hours_to_plot]
lr_stat_matrix_filtered = lr_stat_matrix[:, hours_to_plot]

# Plot p-value heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(p_value_matrix_filtered, 
            xticklabels=hours_to_plot, 
            yticklabels=exog_variables_to_test,
            cmap='RdBu_r',  # Red=high p-value (not sig), Green=low p-value (sig)
            vmin=0, vmax=0.1,  # Focus on significance range
            annot=True, 
            fmt='.2f',
            cbar_kws={'label': 'p-value'},
            linewidths=0.5)
plt.title('LRT p-values by Variable and Hour (α=0.05)')
plt.xlabel('Hour of Day')
plt.ylabel('Exogenous Variable')
plt.tight_layout()
plt.savefig(f"lrt_pvalue_heatmap.pdf")
plt.close()
# plt.savefig('lrt_pvalue_heatmap.png', dpi=300, bbox_inches='tight')
# print("\nHeatmap saved as 'lrt_pvalue_heatmap.png'")
# plt.show()


# Create binary significance heatmap
sig_matrix = (p_value_matrix < 0.05).astype(float)
sig_matrix[np.isnan(p_value_matrix)] = np.nan
sig_matrix_filtered = sig_matrix[:, hours_to_plot]

plt.figure(figsize=(12, 6))
sns.heatmap(sig_matrix_filtered, 
            xticklabels=hours_to_plot, 
            yticklabels=exog_variables_to_test,
            cmap='RdBu',  # Red=not significant, Green=significant
            vmin=0, vmax=1,
            cbar_kws={'label': 'Significant (1=Yes, 0=No)', 'ticks': [0, 1]},
            linewidths=0.5,
            annot=True,
            fmt='.2f')
plt.title('Variable Significance by Hour (α=0.05)')
plt.xlabel('Hour of Day')
plt.ylabel('Exogenous Variable')
plt.tight_layout()
plt.savefig('lrt_significance_heatmap.png', dpi=300, bbox_inches='tight')
print("Heatmap saved as 'lrt_significance_heatmap.png'")
plt.show()

