import pandas as pd
import plotly.express as px
import os
from itertools import product

# --- Mappe med CSV-filer ---
folder_path = "csv_files_copy"  

# --- Tom liste for å samle data fra alle filer ---
all_data = []

# --- Gå gjennom CSV-filer og last inn, rens og legg til state-kolonner ---
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # Konverter kolonner til numerisk og dropp rader med NaN
        df["Distance"] = pd.to_numeric(df["Distance_state"], errors='coerce')
        df["HRV"] = pd.to_numeric(df["HRV_state"], errors='coerce')
        df = df.dropna(subset=["Distance_state", "HRV_state"])

        # --- BEREGN STATES (du kan tilpasse dette etter hva "state" betyr hos deg) ---
        # Eksempel: diskretisering i intervaller eller runde ned
        #df["HRV_state"] = df["HRV"].astype(int)
        #df["Distance_state"] = df["Distance"].astype(int)

        all_data.append(df)

# --- Slå sammen alle filer ---
full_df = pd.concat(all_data, ignore_index=True)

# --- Tell forekomster ---
state_counts = full_df.groupby(["HRV_state", "Distance_state"]).size().reset_index(name="count")

# --- Lag interaktivt bobleplott med Plotly ---
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

# --- Vis og lagre plot ---
fig.show()
fig.write_image("bobleplott_hrv_distance.png")

# --- Finn og skriv ut ubesøkte states ---
possible_states = set(product(range(0, 4), range(0, 6)))  # HRV: 0–3, Distance: 0–5
visited_states = set(zip(state_counts["HRV_state"], state_counts["Distance_state"]))
unvisited_states = sorted(possible_states - visited_states)

print("\n🟥 Ubesøkte tilstander (HRV_state, Distance_state):")
for state in unvisited_states:
    print(state)