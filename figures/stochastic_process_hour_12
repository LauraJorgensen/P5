import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------
# Load data
# ----------------------------------------
df = pd.read_csv("pv_production_june1.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
y = df["pv_production"]

# ----------------------------------------
# Select 7 consecutive days
# ----------------------------------------
unique_days = sorted(list({t.date() for t in y.index}))
selected_days = unique_days[:7]

start = pd.Timestamp(selected_days[0])
end   = pd.Timestamp(selected_days[-1]) + pd.Timedelta(days=1)

y_week = y.loc[start:end]

# ----------------------------------------
# Hourly mean + select hour = 12
# ----------------------------------------
y_hourly = y.resample("H").mean()
y_hour12 = y_hourly[y_hourly.index.hour == 12]   # all days in dataset
y_hour12_week = y_hour12.loc[start:end]          # 7 selected days

# ----------------------------------------
# Plot 1: Weekly PV production with 12:00 markers
# ----------------------------------------
plt.figure(figsize=(10, 3))

# beregn 12:00-tidspunkter og værdier fra den allerede udvalgte serie
twelve_times = y_hour12_week.index
twelve_vals  = y_hour12_week.values

# tegn linjen først

# tegn orange prikker ovenpå linjen (højere zorder så de ligger foran)

plt.plot(y_week.index, y_week.values, linewidth=1, label="PV Production")
plt.scatter(twelve_times, twelve_vals, color='orange', zorder=5, s=50, marker='o', label="Samples at 12:00")


plt.ylim(0, 1)
plt.xlabel("Time")
plt.ylabel("Normalized Production")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------
# Plot 2: Hour-12 stochastic time series
# - All points = black
# - Selected 7 points = orange (uden linjer mellem dem)
# ----------------------------------------

plt.figure(figsize=(10, 3))

# Plot all hour-12 values as black dots
plt.scatter(y_hour12.index, y_hour12.values, s=50, label="Other days", zorder=1)

# Highlight the 7 selected hour-12 values in orange (NO connecting line), draw on top
plt.scatter(y_hour12_week.index, y_hour12_week.values, color="orange", s=50, label="Selected 12:00 samples")

plt.xlabel("Time")
plt.ylabel("PV production")
#plt.title("Stochastic process for hour 12")
plt.tight_layout()
plt.show()
