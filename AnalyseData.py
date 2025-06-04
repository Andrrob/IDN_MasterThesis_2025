import pandas as pd
import os
from scipy.stats import pearsonr

# This script loops through .csv files from data recordings during the experiments and saves a summary of all of them in a new csv file.

# Folder containing the CSV files
folder_path = 
summary_file = 

# List to store results
summary_data = []

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(folder_path, filename)
        
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Ensure numeric conversion (handle possible errors)
        df["Distance"] = pd.to_numeric(df["Distance"], errors='coerce')
        df["HRV"] = pd.to_numeric(df["HRV"], errors='coerce')

        # Drop rows with missing values
        df = df.dropna(subset=["Distance", "HRV"])

        # Extract experiment number and group number (assuming they are in the first row)
        experiment_number = df.iloc[0]["Experiment Number"] if "Experiment Number" in df.columns else "Unknown"
        group_number = df.iloc[0]["Group Number"] if "Group Number" in df.columns else "Unknown"

        # Compute statistics
        max_distance = df["Distance"].max()
        min_distance = df["Distance"].min()
        mean_distance = df["Distance"].mean()
        median_distance = df["Distance"].median()
        var_distance = df["Distance"].var()

        # Compute statistics
        max_hrv = df["HRV"].max()
        min_hrv = df["HRV"].min()
        mean_hrv = df["HRV"].mean()
        median_hrv = df["HRV"].median()
        var_hrv = df["HRV"].var()

        # Calculate correlation
        if len(df) > 1:  # Ensure at least 2 data points for correlation
            correlation, p_value = pearsonr(df["Distance"], df["HRV"])
        else:
            correlation, p_value = None, None  # Not enough data

        # Append results to the list
        summary_data.append([
            experiment_number, group_number,
            max_distance, min_distance, mean_distance, median_distance, var_distance, max_hrv, min_hrv, mean_hrv, median_hrv, var_hrv,
            correlation, p_value
        ])

# Create a DataFrame for summary
summary_df = pd.DataFrame(summary_data, columns=[
    "Experiment Number", "Group Number",
    "Max Distance", "Min Distance", "Mean Distance", "Median Distance", "var_distance", "max_hrv", "min_hrv", "mean_hrv", "median_hrv", "var_hrv",
    "Correlation (Distance-HRV)", "P-value"
])

# Save summary to CSV
summary_df.to_csv(summary_file, index=False)

print(f"Summary saved to {summary_file}")
