import pandas as pd
import matplotlib.pyplot as plt

# hent data
df = pd.read_csv("data/pv_production_june_clean.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
y = df["pv_production"]

# vælg 7 dage til markering
unique_days = sorted(list({t.date() for t in y.index}))
selected_days = unique_days[:7]

start = pd.Timestamp(selected_days[0])
end   = pd.Timestamp(selected_days[-1]) + pd.Timedelta(days=1)

y_week = y.loc[start:end]

# få time-12 værdier
y_hourly = y.resample("H").mean()
y_hour12 = y_hourly[y_hourly.index.hour == 12]   
y_hour12_week = y_hour12.loc[start:end]         


plt.figure(figsize=(10, 3))

# beregn 12:00-tidspunkter og værdier fra den allerede udvalgte serie
twelve_times = y_hour12_week.index
twelve_vals  = y_hour12_week.values


# tegn orange prikker ovenpå linjen 

plt.plot(y_week.index, y_week.values, linewidth=1, label="PV Production")
plt.scatter(twelve_times, twelve_vals, color='orange', zorder=5, s=50, marker='o', label="Samples at 12:00")


plt.ylim(0, 1)
plt.xlabel("Time")
plt.ylabel("Normalized Production")
plt.legend()
plt.tight_layout()
plt.show()

# Tegn kun time-12 værdierne, fremhæv de 7 udvalgte dage

plt.figure(figsize=(10, 3))
plt.scatter(y_hour12.index, y_hour12.values, s=50, label="Other days", zorder=1)
plt.scatter(y_hour12_week.index, y_hour12_week.values, color="orange", s=50, label="Selected 12:00 samples")
plt.xlabel("Time")
plt.ylabel("PV production")
plt.tight_layout()
plt.show()
