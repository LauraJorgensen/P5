import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pvlib

# --- Load PV production data ---
data = pd.read_csv('data/pv_production_june_clean.csv', parse_dates=['timestamp'])
P_full = data['pv_production'].values
times = pd.DatetimeIndex(data['timestamp'])

# --- Calculate poa_baseline (b_t) ---
latitude = 48.669
longitude = 12.692
tz = 'Europe/Berlin'
site = pvlib.location.Location(latitude, longitude, tz=tz)
tilt = latitude
azimuth = 180
albedo = 0.2
clearsky = site.get_clearsky(times)
solpos = site.get_solarposition(times)
poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    solar_zenith=solpos['apparent_zenith'],
    solar_azimuth=solpos['azimuth'],
    dni=clearsky['dni'],
    ghi=clearsky['ghi'],
    dhi=clearsky['dhi'],
    albedo=albedo
)
b_full = poa['poa_global'].values
b_full = b_full / np.max(b_full)  # Normalize

# --- Filter for P_t > 0 and b_t > 0 ---
mask = (P_full > 0) & (b_full > 0)
P = P_full[mask]
b = b_full[mask]

# --- Divide data by normalized baseline ---
# Handle division by zero or invalid values
normalized_ratio = np.where(b_full > 0, P_full / b_full, np.nan)

# --- Create plot ---
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(times, normalized_ratio, color='green', alpha=0.7)
plt.xlabel('Time', fontsize=11)
plt.ylabel('Normalized Ratio', fontsize=11)
plt.title('PV Production Divided by Normalized Baseline', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('normalized_ratio.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'normalized_ratio.png'")
plt.show()
