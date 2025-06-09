import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

"""
This script creates visualizations for distance data across three groups:
1. Mean distance with confidence intervals
2. Histograms with KDE for each group
3. QQ-plots for normality checking
4. Boxplots of distance per group
The visualizations are saved as PNG files.
"""

# Load data
summary_df = pd.read_csv("distance_hrv_summary.csv")

# Extract group data for plotting
group1 = summary_df[summary_df["Group Number"] == 1]["Mean Distance"].dropna()
group2 = summary_df[summary_df["Group Number"] == 2]["Mean Distance"].dropna()
group3 = summary_df[summary_df["Group Number"] == 3]["Mean Distance"].dropna()

def plotting():
    # === Visualizations ===
    # --- 1. Mean distance with confidence intervals ---
    plt.figure(figsize=(10, 6))
    for idx, group_data in enumerate([group1, group2, group3], start=1):
        n = len(group_data)
        mean = np.mean(group_data)
        std = np.std(group_data, ddof=1)  # Use ddof=1 for sample standard deviation
        
        # Calculate 95% confidence interval using the t-distribution
        ci_lower, ci_upper = t.interval(
            confidence=0.95,
            df=n-1,  # Degrees of freedom = number of data points - 1
            loc=mean,  
            scale=std / np.sqrt(n)  # Standard error = std / sqrt(n)
        )
        
          # Plot with error bars
        plt.errorbar(
            x=[idx],
            y=[mean],
            yerr=[[mean - ci_lower], [ci_upper - mean]],
            fmt='o',
            color=['blue', 'green', 'red'][idx-1],
            capsize=5,
            elinewidth=2,
            markersize=8,
            label=f"Gruppe {idx}"
        )

    plt.title("Mean Distance with 95% Confidence Intervals")
    plt.ylabel("Mean Distance")
    plt.xticks([1, 2, 3], ['Group 1', 'Group 2', 'Group 3'])
    plt.grid(axis='y')
    plt.legend()

    plt.tight_layout()
    plt.savefig("ConfidenceInterval_meanDistance.png", dpi=300)
    plt.show()

    # --- 2. Histograms with KDE ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.histplot(group1, kde=True, ax=axes[0], color="blue")
    axes[0].set_title("Group 1: Histogram & KDE")
    axes[0].set_xlabel("Mean Distance")
    axes[0].set_ylabel("Density")

    sns.histplot(group2, kde=True, ax=axes[1], color="green")
    axes[1].set_title("Group 2: Histogram & KDE")
    axes[1].set_xlabel("Mean Distance")
    axes[1].set_ylabel("Density")

    sns.histplot(group3, kde=True, ax=axes[2], color="red")
    axes[2].set_title("Group 3: Histogram & KDE")
    axes[2].set_xlabel("Mean Distance")
    axes[2].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig("Histogram_KDE.png", dpi=300)
    plt.show()

    # --- 3. QQ-plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, label in zip(axes, [group1, group2, group3], ["Group 1", "Group 2", "Group 3"]):
        sm.qqplot(data, line="s", ax=ax)
        ax.set_title(f"QQ-plot: {label}")
    plt.tight_layout()
    plt.savefig("QQplots.png", dpi=300)
    plt.show()

    # --- 4. Boxplot ---
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="Group Number", y="Mean Distance", data=summary_df, hue="Group Number", palette="Set2", legend=False)
    plt.title("Boxplot of Mean Distance per Group")
    plt.xlabel("Group")
    plt.ylabel("Mean Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("boxplot_mean_distance.png", dpi=300)
    plt.show()

# Run the plotting function
plotting()
