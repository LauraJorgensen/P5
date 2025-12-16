import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib

data = pd.read_csv('data/pv_production_june_clean.csv', parse_dates=['timestamp'])
P_full = data['pv_production'].values
times = pd.DatetimeIndex(data['timestamp'])

# --- Clear sky model ---
latitude = 48.669
longitude = 12.692
tz = 'Europe/Berlin'
site = pvlib.location.Location(latitude, longitude, tz=tz)
tilt = 20
azimuth = 180
clearsky = site.get_clearsky(times)
solpos = site.get_solarposition(times)
poa = pvlib.irradiance.get_total_irradiance(surface_tilt=tilt,surface_azimuth=azimuth,solar_zenith=solpos['apparent_zenith'],solar_azimuth=solpos['azimuth'],dni=clearsky['dni'],ghi=clearsky['ghi'],dhi=clearsky['dhi'],)
b_full = poa['poa_global'].values
b_full = b_full / np.max(b_full)

# --- Compute ratio between clear sky and data ---
normalized_ratio = np.where(b_full > 0, P_full / b_full, np.nan)

# --- Plot ---
plt.figure(figsize=(10, 4))
plt.plot(times, normalized_ratio)
plt.xlabel('Time')
plt.ylabel('PV Production / $B_t$')
plt.grid(True)
plt.xlim([times.min(), times.max()])
plt.ylim([0, 30])
plt.tight_layout()
plt.savefig(f"normalized_ratio.pdf")
plt.close()
