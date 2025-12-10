import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
from statsmodels.graphics.tsaplots import plot_acf
from scipy.signal import welch

# --- Konfiguration ---
latitude, longitude = 48.6727, 12.6931  # Landau a. d. Isar

# --- Generér tid for et helt år (15-min interval) ---
year = 2022
times = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:45', freq='15min', tz='Europe/Berlin')

# --- Beregn clearsky ---
location = pvlib.location.Location(latitude, longitude)
clearsky = location.get_clearsky(times)
Bt = clearsky['ghi'].values

# --- Tilføj støj, men kun hvis clearsky > 0.1 ---
noise = np.random.normal(-200, 0.10 * np.max(Bt), size=Bt.shape)
Bt_noisy = np.where(Bt >= 0.1, Bt + noise, Bt)
Bt_noisy = np.maximum(Bt_noisy, 0)

# --- Normaliser ---
Bt_noisy_norm = Bt_noisy / np.max(Bt_noisy)

# --- Udglat med dagligt gennemsnit ---
df_plot = pd.DataFrame({'norm_prod': Bt_noisy_norm}, index=times)
df_daily = df_plot.resample('D').mean()

# --- Plot med høj kvalitet ---
plt.figure(figsize=(15, 5), dpi=300)
plt.plot(times, Bt_noisy_norm, color='tab:blue', linewidth=0.7, label='Normalized Power Production')
plt.ylim(0, 1.05)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Normalized Production', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.legend(fontsize=14)
plt.savefig('clearsky_noise_highres_norm.pdf', format='pdf', dpi=300)
plt.savefig('clearsky_noise_highres_norm.png', format='png', dpi=300)
plt.show()

# --- Indlæs data ---
df = pd.read_csv('data/pv_production_june_clean.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# --- Udvælg sidste uge ---
last_week_start = df.index[-1] - pd.Timedelta(days=7)
df_last_week = df[df.index >= last_week_start]

# --- Normaliser produktion ---
pv_max = df['pv_production'].max()
df_last_week['pv_production_norm'] = df_last_week['pv_production'] / pv_max

# --- Plot ---
plt.figure(figsize=(12, 4), dpi=300)
plt.plot(df_last_week.index, df_last_week['pv_production_norm'], color='tab:orange', linewidth=1.2, label='Normalized PV Production')
plt.ylim(0, 1.05)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Normalized Production', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.legend(fontsize=14)
plt.savefig('pv_last_week_highres.pdf', format='pdf', dpi=300)
plt.savefig('pv_last_week_highres.png', format='png', dpi=300)
plt.show()

# --- Plot ACF for all data ---
plt.figure(figsize=(8, 4), dpi=300)
plot_acf(df['pv_production'], lags=200, ax=plt.gca(),alpha=None)
plt.title('Autocorrelation (ACF) of PV Production', fontsize=14)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Autocorrelation', fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig('pv_acf_highres.pdf', format='pdf', dpi=300)
plt.savefig('pv_acf_highres.png', format='png', dpi=300)
plt.show()

# --- Plot PSD for all data ---
fs = 4  # 15-min data = 4 samples per hour
f, Pxx = welch(df['pv_production'].values, fs=fs, nperseg=1024)
plt.figure(figsize=(8, 4), dpi=300)
plt.semilogy(f, Pxx, color='tab:green')
plt.title('Power Spectral Density (PSD) of PV Production', fontsize=14)
plt.xlabel('Frequency [1/hour]', fontsize=12)
plt.ylabel('PSD', fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig('pv_psd_highres.pdf', format='pdf', dpi=300)
plt.savefig('pv_psd_highres.png', format='png', dpi=300)
plt.show()