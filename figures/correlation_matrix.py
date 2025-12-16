import pandas as pd
import numpy as np
import requests
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = 'data/pv_production_june_clean.csv'
target_col = 'pv_production'
latitude, longitude = 48.6727, 12.6931
period = 96

df = pd.read_csv(csv_path, parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
y = df[target_col].astype(float)

# --- hent vejrdata --- 
def fetch_open_meteo_data(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        print("Error fetching Open-Meteo data:", r.status_code)
        return None

pv_times = df.index.strftime('%Y-%m-%dT%H:%M')
start_date = pv_times[0][:10]
end_date   = pv_times[-1][:10]

weather_url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    f"&hourly=temperature_2m,cloud_cover,direct_radiation,"
    f"diffuse_radiation,cloudcover_low,cloudcover_mid,cloudcover_high"
    f"&timezone=Europe/Berlin"
)

weather_data = fetch_open_meteo_data(weather_url)

if weather_data:
    hourly = weather_data.get('hourly', {})
    weather_times = hourly.get('time', [])

    weather_df = pd.DataFrame({
        "time": pd.to_datetime(weather_times),
        "temperature_2m": hourly.get("temperature_2m"),
        "cloud_cover": hourly.get("cloud_cover"),
        "direct_radiation": hourly.get("direct_radiation"),
        "diffuse_radiation": hourly.get("diffuse_radiation"),
        "cloudcover_low": hourly.get("cloudcover_low"),
        "cloudcover_mid": hourly.get("cloudcover_mid"),
        "cloudcover_high": hourly.get("cloudcover_high")
    }).set_index("time")
    weather_on_pv = weather_df.reindex(df.index, method="nearest")
else:
    weather_on_pv = pd.DataFrame(index=df.index)

# --- Correlation matrix --- 
mask = np.triu(np.ones_like(corr, dtype=bool), k=0) 
plt.figure(figsize=(10, 8))
sns.set_style("white")
ax = sns.heatmap(corr,mask=~mask,cmap='RdBu',vmin=-1, vmax=1,annot=True,fmt=".2f",linewidths=0.6,linecolor="white",cbar_kws={"label": "Corr"},square=True)
plt.xticks(rotation=50, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"corr_matrix.pdf")
plt.close()
