import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.stats import chi2
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# --- Konfiguration ---
csv_path = 'data/pv_production_june_clean.csv'
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

# split i træning og validering
val_h = int(period * 1.5) 

X_train, X_val = X.iloc[:-val_h, :], X.iloc[-val_h:, :]
y_train, y_val = y.iloc[:-val_h], y.iloc[-val_h:]

#  resample til hourly data
y_hour = y.resample('H').mean()
X_hour = X.resample('H').mean()

val_h_hours = int(val_h / 4)
y_train_hour, y_val_hour = y_hour.iloc[:-val_h_hours], y_hour.iloc[-val_h_hours:]
X_train_hour = X_hour.iloc[:-val_h_hours, :]
X_val_hour = X_hour.iloc[-val_h_hours:, :]


exog_variables_to_test = ['cloud_cover', 'temperature', 'direct_radiation', 
                          'diffuse_radiation', 'cloudcover_low', 'cloudcover_mid', 
                          'cloudcover_high']

exog_variables_to_test = [v for v in exog_variables_to_test if v in X.columns]

print("LRT for exogenous variables:")
print("-" * 40)


sarima_orders_by_hour = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (1, 0, 3),
    4: (1, 0, 2),
    5: (0, 1, 1),
    6: (0, 1, 1),
    7: (2, 1, 0),
    8: (0, 1, 1),
    9: (0, 1, 1),
    10: (0, 1, 1),
    11: (0, 1, 2),
    12: (2, 1, 1),
    13: (2, 1, 0),
    14: (2, 1, 0),
    15: (2, 1, 0),
    16: (0, 1, 1),
    17: (3, 0, 0),
    18: (1, 0, 2),
    19: (2, 0, 0),
    20: (0, 0, 0),
    21: (0, 0, 0),
    22: (0, 0, 0),
    23: (0, 0, 0),
}


min_samples_per_hour = 14
lrt_results_by_var = {var: {} for var in exog_variables_to_test}

for var in exog_variables_to_test:
    print(f"\nTesting variable: {var}")
    print("-" * 40)
    
    for h in range(24):
        idx_train_h = y_train_hour.index[y_train_hour.index.hour == h]
        y_train_h = y_train_hour.loc[idx_train_h].dropna()
        p, d, q = sarima_orders_by_hour[h]
        if len(y_train_h) < min_samples_per_hour:
            lrt_results_by_var[var][h] = None
            continue
        
        X_train_h_single = X_train_hour.loc[idx_train_h, [var]].reindex(y_train_h.index).ffill()
        
        # Fit model uden exogenous variable
        model_null = SARIMAX(endog=y_train_h,order=(p, d, q),enforce_stationarity=True,enforce_invertibility=True,trend='n')
        res_null = model_null.fit(disp=False, maxiter=100)
        ll_null = res_null.llf
        
        # Fit model med exogenous variable
        model_alt = SARIMAX(endog=y_train_h,exog=X_train_h_single,order=(p, d, q),enforce_stationarity=True,enforce_invertibility=True,trend='n')
        res_alt = model_alt.fit(disp=False, maxiter=100)
        ll_alt = res_alt.llf

        # Likelihood ratio test statistic
        lr_stat = 2 * (ll_alt - ll_null)
        df_diff = X_train_h_single.shape[1]  
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
        print(f"  Significant at alpha=0.05 for hours: {significant_hours}")
    else:
        print(f"  Not significant for any hour")

# Print detailed LRT results
print("LRT results")
print("-" * 40)

for var in exog_variables_to_test:
    print(f"\n{var}:")
    print(f"{'Hour':<6} {'LR Stat':<10} {'p-value':<10} {'Significant':<12}")
    print("-" * 40)
    for h in range(24):
        result = lrt_results_by_var[var].get(h)
        sig_mark = "Yes" if result['significant'] else "No"
        print(f"{h:<6} {result['lr_stat']:<10.3f} {result['p_value']:<10.4f} {sig_mark:<12}")

# Heatmap 
import seaborn as sns
p_value_matrix = np.zeros((len(exog_variables_to_test), 24))
lr_stat_matrix = np.zeros((len(exog_variables_to_test), 24))

for i, var in enumerate(exog_variables_to_test):
    for h in range(24):
        result = lrt_results_by_var[var].get(h)

        p_value_matrix[i, h] = result['p_value']
        lr_stat_matrix[i, h] = result['lr_stat']
    
# remove 0,1,2,20,21,22,23 
hours_to_plot = list(range(3, 20))
p_value_matrix_filtered = p_value_matrix[:, hours_to_plot]
lr_stat_matrix_filtered = lr_stat_matrix[:, hours_to_plot]

# Plot p-value heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(p_value_matrix_filtered, 
            xticklabels=hours_to_plot, 
            yticklabels=exog_variables_to_test,
            cmap='RdBu_r',  
            vmin=0, vmax=0.1,  
            annot=True, 
            fmt='.2f',
            cbar_kws={'label': '$p$-value'},
            linewidths=0.5)
plt.xlabel('Hour')
plt.ylabel('Exogenous Variable')
plt.tight_layout()
plt.savefig(f"lrt_pvalue_heatmap.pdf")
plt.close()
