import pandas as pd
import numpy as np
from scipy.stats import t, levene, bartlett, shapiro, ttest_ind, mannwhitneyu
import itertools

# This script performs statistical comparisons between groups in the dataset "summary.csv" created in AnalyzeData.py.
# It calculates confidence intervals, tests for normality and variance homogeneity, 
# chooses appropriate statistical tests (T-test or Mann-Whitney U-test), 
# and exports the groupwise comparison results to an Excel file.

summary_df = pd.read_csv("summary.csv")
labels = [("Mean Distance", "mean"), ("Median Distance", "median"), ("Min Distance", "min")]
#labels = [("Mean Distance"), ("Median Distance"), ("Min Distance")]
groups = sorted(summary_df["Group Number"].unique())

def confidence_interval(data, confidence=0.95, mode="mean"):
    n = len(data)
    std = data.std(ddof=1)
    t_crit = t.ppf((1 + confidence) / 2., n - 1)
    margin_of_error = t_crit * std / np.sqrt(n)

    if mode == "mean":
        center = data.mean()
    elif mode == "median":
        center = data.median()
    elif mode == "min":
        center = data.min()
    else:
        raise ValueError("Mode must be 'mean', 'median', or 'min'")

    lower = center - margin_of_error
    upper = center + margin_of_error
    return center, std, n, lower, upper

def build_comparison_table(g1, g2):
    rows = []

    for col_name, label in labels:
        d1 = summary_df[summary_df["Group Number"] == g1][col_name].dropna()
        d2 = summary_df[summary_df["Group Number"] == g2][col_name].dropna()

        # Variance tests
        bartlett_stat, bartlett_p = bartlett(d1, d2)
        stat_lev, p_lev = levene(d1, d2)
        # Normality tests
        SW_p_d1, res_SW_1 = Shapiro_Wilk_test(d1)
        SW_p_d2, res_SW_2 = Shapiro_Wilk_test(d2)

        if res_SW_1==True and res_SW_2==True and bartlett_p > 0.05:
            # T-test
            t_stat, p_t = ttest_ind(d1, d2, equal_var=True)
            if t_stat < 0:
                direction = f"Group {g1} < Group {g2}"
            else:
                direction = f"Group {g1} ≥ Group {g2}"
            #direction = "g1 < g2" if t_stat < 0 else "g1 ≥ g2"
            significant = "Yes" if t_stat < 0 and p_t< 0.05 else "No"
            rows.append({
            "Label": label,
            "Levene p": round(p_lev, 4),
            "Bartlett p": round(bartlett_p, 4),
            "Test": "T-test",
            "Test stat": round(t_stat, 4),
            "Two-tailed p": round(p_t, 4),
            "Result": direction,
            "Significant?": significant
            })
        else:
            # Mann-Whitney U-test
            u_stat, p_u = mannwhitneyu(d2, d1, alternative='two-sided')  # Tester om d1 < d2
            if p_u < 0.05:
                direction = f"Group {g1} < Group {g2}"
                significant = "Yes"
            else:
                direction = f"Group {g1} ≥ Group {g2}"
                significant = "No"
            rows.append({
                "Label": label,
                "Levene p": round(p_lev, 4),
                "Bartlett p": round(bartlett_p, 4),#
                "Test": "Mann-Whitney U-test",
                "Test stat": round(u_stat, 4),
                "Two-tailed p": round(p_u, 4),
                "Result": direction,
                "Significant?": significant
            })

    return pd.DataFrame(rows)

def run_and_export_groupwise_tables():
    with pd.ExcelWriter("appendix_groupwise_tests.xlsx", engine="xlsxwriter") as writer:
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                df = build_comparison_table(g1, g2)
                sheet_name = f"Group {g1} vs {g2}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("[✓] Groupwise test results saved to appendix_groupwise_tests.xlsx")

def Shapiro_Wilk_test(data):
    stat, p = shapiro(data)
    #print(f"Group {group}: p = {p:.4f}", end=" → ")
    if p < 0.05:
        #print("❗ Significantly deviates from normal distribution")
        return p, False
    else:
        #print("✅ Can assume normal distribution")
        return p, True

def test_normal_dist():
    for col_name, label in labels:
        print(f"\n=== Normality test for each group (Shapiro-Wilk) {col_name} ===")
        #group1 = summary_df[summary_df["Group Number"] == 1][label].dropna()
        #group2 = summary_df[summary_df["Group Number"] == 2][label].dropna()
        #group3 = summary_df[summary_df["Group Number"] == 3][label].dropna()
    
        for group in sorted(summary_df["Group Number"].unique()):
            group_data = summary_df[summary_df["Group Number"] == group][col_name].dropna()
            p, res = Shapiro_Wilk_test(group_data)
            print(f"Group {group}: p = {p:.4f}", end=" → ")
            if res == True:
                print("✅ Can assume normal distribution")
            else: 
                print("❗ Significantly deviates from normal distribution")

def print_basic_res():
    label = "Min Distance"
    group1 = summary_df[summary_df["Group Number"] == 1][label].dropna()
    group2 = summary_df[summary_df["Group Number"] == 2][label].dropna()
    group3 = summary_df[summary_df["Group Number"] == 3][label].dropna()

    print(f"Average min distance in group 1: {np.mean(group1)}")
    print(f"Average min distance in group 2: {np.mean(group2)}")
    print(f"Average min distance in group 3: {np.mean(group3)}")

    print(f"Median of min distance in group 1: {np.median(group1)}")
    print(f"Median of min distance in group 2: {np.median(group2)}")
    print(f"Median of min distance in group 3: {np.median(group3)}")



#if __name__ == "__main__":
#    run_and_export_groupwise_tables()
