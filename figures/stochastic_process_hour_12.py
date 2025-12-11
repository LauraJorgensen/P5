import pandas as pd
import matplotlib.pyplot as plt

# hent data
df = pd.read_csv("data/pv_production_june_clean.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
y = df["pv_production"]
times = pd.DatetimeIndex(data['timestamp'])

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


# plot 1
plt.figure(figsize=(10, 4))
twelve_times = y_hour12_week.index
twelve_vals  = y_hour12_week.values
plt.plot(y_week.index,y_week.values,linewidth=1.2,color="tab:blue",label="Normalized PV Production")
plt.scatter(twelve_times,twelve_vals,color="tab:orange",s=40,marker='o',zorder=5,label="Samples at 12:00")
plt.ylim(0, 1)
plt.xlim(start, end)
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Normalized PV Production")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"plot1_kl12.pdf")
plt.close()

# plot 2
plt.figure(figsize=(10, 4))
plt.scatter(y_hour12.index,y_hour12.values,s=40,alpha=0.6,)
plt.scatter(y_hour12_week.index, y_hour12_week.values, color="tab:orange", s=40, zorder=5)
plt.grid(True)
plt.xlabel("Time")
plt.ylim(0, 1)
plt.ylabel("Normalized PV Production")
plt.xlim(times[0], times[-1])
plt.tight_layout()
plt.savefig(f"plot2_kl12.pdf")
plt.close()
