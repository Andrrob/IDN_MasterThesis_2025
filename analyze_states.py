import pandas as pd
import plotly.express as px
import os
from itertools import product


# This script processes CSV files from experiments, summarizes state occurrences, and visualizes them in a bubble plot.

# --- Folder containing CSV files ---
folder_path = "csv_files_copy"  

# --- Empty list to collect data from all files ---
all_data = []


# --- Loop through CSV files, load, clean, and add state columns ---
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # --- Convert columns to numeric and drop rows with NaN ---
        df["Distance"] = pd.to_numeric(df["Distance_state"], errors='coerce')
        df["HRV"] = pd.to_numeric(df["HRV_state"], errors='coerce')
        df = df.dropna(subset=["Distance_state", "HRV_state"])

        all_data.append(df)

# --- Combine all files ---
full_df = pd.concat(all_data, ignore_index=True)

# --- Count occurrences ---
state_counts = full_df.groupby(["HRV_state", "Distance_state"]).size().reset_index(name="count")

# --- Create interactive bubble plot with Plotly ---
fig = px.scatter(
    state_counts,
    x="HRV_state",
    y="Distance_state",
    size="count",
    size_max=40,
    title="State Exploration Frequency",
    labels={"HRV_state": "HRV state", "Distance_state": "Distance state"},
    hover_name="count"
)

fig.update_layout(
    xaxis=dict(dtick=1, tickmode="linear", range=[-0.5, 3.5]),
    yaxis=dict(dtick=1, tickmode="linear", range=[-0.5, 5.5]),
    plot_bgcolor="white"
)

# --- Show and save plot ---
fig.show()
fig.write_image("bobleplott_hrv_distance.png")

# --- Find and print unvisited states ---
possible_states = set(product(range(0, 4), range(0, 6)))  # HRV: 0–3, Distance: 0–5
visited_states = set(zip(state_counts["HRV_state"], state_counts["Distance_state"]))
unvisited_states = sorted(possible_states - visited_states)

print("\n🟥 Unvisited states (HRV_state, Distance_state):")
for state in unvisited_states:
    print(state)
