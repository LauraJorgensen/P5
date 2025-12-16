import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
from statsmodels.graphics.tsaplots import plot_acf
from scipy.signal import welch

np.random.seed(42)

# --- Konfiguration ---
latitude, longitude = 48.6727, 12.6931  # Landau a. d. Isar
year = 2025
times = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:45', freq='15min', tz='Europe/Berlin')

df = pd.read_csv('data/pv_production_june_clean.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# --- Simuler årlig solcelleproduktion --- 
location = pvlib.location.Location(latitude, longitude)
clearsky = location.get_clearsky(times)
Bt = clearsky['ghi'].values

noise = np.random.normal(-200, 0.10 * np.max(Bt), size=Bt.shape)
Bt_noisy = np.where(Bt >= 0.1, Bt + noise, Bt)
Bt_noisy = np.maximum(Bt_noisy, 0)
Bt_noisy_norm = Bt_noisy / np.max(Bt_noisy + 225)

# --- Plot normeret clearsky + noise ---
plt.figure(figsize=(10, 4))
plt.plot(times, Bt_noisy_norm, linewidth=1)
plt.ylim(0, 1)
plt.xlim(times[0], times[-1])
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('Normalized PV Production')
plt.tight_layout()
plt.savefig('clearsky_noise_highres_norm.pdf')
plt.close()

# --- Udvælg og plot sidste uge i dataset ---
last_week_start = df.index[-1] - pd.Timedelta(days=7)
df_last_week = df[df.index >= last_week_start]

plt.figure(figsize=(10, 4))
plt.plot(df_last_week.index, df_last_week['pv_production'], linewidth=1.2, label='PV Production')
plt.ylim(0, 1)
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('Normalized PV Production')
plt.tight_layout()
plt.savefig('pv_last_week_highres.pdf')
plt.close()

# --- Plot ACS for hele dataset---
plt.figure(figsize=(6, 4))
ax = plt.gca()
plot_acf(df['pv_production'], lags=200, ax=ax, alpha=None)
ax.set_title("")  
plt.xlabel('Lag')
plt.xlim(0, 200)
plt.ylabel('ACS')
plt.grid(True)
plt.tight_layout()
plt.savefig('pv_acf_highres.pdf')
plt.close()

# --- Plot PSD for all data ---
fs = 4  # 4 samples per hour (15-min data)
f, Pxx = welch(df['pv_production'].values, fs=fs, nperseg=1024)

plt.figure(figsize=(6, 4))
plt.semilogy(f, Pxx)
plt.xlabel('Frequency [1/hour]')
plt.ylabel('Periodgram')
plt.xlim(0, 2)
plt.grid(True)
plt.tight_layout()
plt.savefig('pv_psd_highres.pdf')
plt.close()

# --- Plot PSD kun for time 12 ---
df_hour12 = df[df.index.hour == 12]
f_h12, Pxx_h12 = welch(df_hour12['pv_production'].values, fs=fs,nperseg=min(256, len(df_hour12)))
plt.figure(figsize=(6, 4))
plt.semilogy(f_h12, Pxx_h12)
plt.xlabel('Frequency [1/hour]')
plt.ylabel('PSD')
plt.xlim(0, 2)
plt.grid(True)
plt.tight_layout()
plt.savefig('pv_psd_hour12_highres.pdf')
plt.close()
