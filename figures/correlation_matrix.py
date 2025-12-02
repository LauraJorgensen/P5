import pandas as pd
import numpy as np
import requests
import seaborn as sns
import matplotlib.pyplot as plt

# ================================================
# LOAD PV DATA
# ================================================

df = pd.read_csv(csv_path, parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# Convert target to float in case it's stored as string
y = df[target_col].astype(float)

# ================================================
# FUNCTION TO FETCH WEATHER DATA
# ================================================

def fetch_open_meteo_data(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        print("Error fetching Open-Meteo data:", r.status_code)
        return None

# ================================================
# DETERMINE DATE RANGE
# ================================================

pv_times = df.index.strftime('%Y-%m-%dT%H:%M')
start_date = pv_times[0][:10]
end_date   = pv_times[-1][:10]

# ================================================
# OPEN-METEO URL
# ================================================

weather_url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    f"&hourly=temperature_2m,cloud_cover,direct_radiation,"
    f"diffuse_radiation,cloudcover_low,cloudcover_mid,cloudcover_high"
    f"&timezone=Europe/Berlin"
)

print("Fetching weather data from:")
print(weather_url)

# ================================================
# FETCH WEATHER
# ================================================

weather_data = fetch_open_meteo_data(weather_url)

if weather_data:

    hourly = weather_data.get('hourly', {})
    weather_times = hourly.get('time', [])

    # safe extraction with fallback
    def safe(key):
        return hourly.get(key, [np.nan] * len(weather_times))

    weather_df = pd.DataFrame({
        "time":              pd.to_datetime(weather_times),
        "temperature_2m":    safe("temperature_2m"),
        "cloud_cover":       safe("cloud_cover"),
        "direct_radiation":  safe("direct_radiation"),
        "diffuse_radiation": safe("diffuse_radiation"),
        "cloudcover_low":    safe("cloudcover_low"),
        "cloudcover_mid":    safe("cloudcover_mid"),
        "cloudcover_high":   safe("cloudcover_high")
    }).set_index("time")

    # align timestamps
    weather_on_pv = weather_df.reindex(df.index, method="nearest")

else:
    print("No weather data fetched.")
    weather_on_pv = pd.DataFrame(index=df.index)

# ================================================
# CORRELATION MATRIX
# ================================================

X = weather_on_pv.copy()
corr = X.corr()

print("\n=== Correlation Matrix ===\n")
print(corr.round(3))

# ================================================
# HEATMAP (upper triangular)
# ================================================

mask = np.triu(np.ones_like(corr, dtype=bool), k=0)  # keep diagonal + upper

plt.figure(figsize=(12, 10))
sns.set_style("white")

# smooth diverging palette similar to your example
#cmap = sns.diverging_palette(10, 220, as_cmap=True)

ax = sns.heatmap(
    corr,
    mask=~mask,          # show ONLY upper triangle
    cmap='RdBu',
    vmin=-1, vmax=1,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 12},
    linewidths=0.6,
    linecolor="white",
    cbar_kws={"label": "Corr"},
    square=True
)

plt.xticks(rotation=50, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.title("Correlation Matrix of Exogenous Weather Variables", fontsize=16, pad=20)
plt.tight_layout()
plt.show()
