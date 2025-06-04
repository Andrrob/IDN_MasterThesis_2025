import pandas as pd

import numpy as np
from scipy.stats import t, levene, bartlett, shapiro, anderson, ttest_ind, f_oneway

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# === Last inn data ===

summary_df = pd.read_csv("distance_hrv_summary.csv")
# === Funksjon for å beregne konfidensintervall for gjennomsnitt ===
def confidence_interval_mean(data, confidence=0.95):
    n = len(data)
    mean = data.mean()
    std = data.std(ddof=1)
    t_crit = t.ppf((1 + confidence) / 2., n - 1)
    margin_of_error = t_crit * std / np.sqrt(n)
    lower = mean - margin_of_error
    upper = mean + margin_of_error
    return mean, std, n, lower, upper

def StatisticalHypothesisTestMean():
    # === Steg 1 & 2: Beregn estimater og CI for hver gruppe ===
    results = []
    for group in sorted(summary_df["Group Number"].unique()):
        group_data = summary_df[summary_df["Group Number"] == group]["Mean Distance"].dropna()
        
        mean, std, n, ci_lower, ci_upper = confidence_interval_mean(group_data)
        
        results.append({
            "Group": group,
            "N": n,
            "Mean Distance": mean,
            "Standard Deviation": std,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper
        })

    # === Lag en ny DataFrame med resultatene ===
    results_df = pd.DataFrame(results)

    # === Vis og/eller lagre resultatene ===
    print(results_df.to_string(index=False))

    # Du kan lagre det til CSV hvis ønskelig
    results_df.to_csv("group_distance_CI_summary.csv", index=False)

    # === Ekstra: Test likhet i varians mellom gruppe 1 og gruppe 3 ===
    group1 = summary_df[summary_df["Group Number"] == 1]["Mean Distance"].dropna()
    group2 = summary_df[summary_df["Group Number"] == 2]["Mean Distance"].dropna()
    group3 = summary_df[summary_df["Group Number"] == 3]["Mean Distance"].dropna()

    # Levene's test (robust for ulik fordeling)
    stat_lev, p_lev = levene(group1, group3)
    print(f"\nLevene's test for lik varians: p = {p_lev:.4f}")

    # Bartlett's test (forutsetter normalfordeling)
    stat_bart, p_bart = bartlett(group1, group3)
    print(f"Bartlett's test for lik varians: p = {p_bart:.4f}")

    # Tolkning
    if p_lev < 0.05 or p_bart < 0.05:
        print("❗ Det er signifikant forskjell i varians mellom gruppe 1 og 3.")
    else:
        print("✅ Ingen signifikant forskjell i varians mellom gruppe 1 og 3.")


    # === Normalitetstester: Shapiro-Wilk ===
    print("\n=== Test av normalfordeling for hver gruppe (Shapiro-Wilk) ===")
    for group in sorted(summary_df["Group Number"].unique()):
        group_data = summary_df[summary_df["Group Number"] == group]["Mean Distance"].dropna()
        stat, p = shapiro(group_data)
        print(f"Gruppe {group}: p = {p:.4f}", end=" → ")
        if p < 0.05:
            print("❗ Avviker signifikant fra normalfordeling")
        else:
            print("✅ Kan anta normalfordeling")

    # === Normalitetstester: Anderson-Darling teststatistikk ===
    """"
    for group in sorted(summary_df["Group Number"].unique()):
        group_data = summary_df[summary_df["Group Number"] == group]["Mean Distance"].dropna()
        result = anderson(group_data)
        print(f"Anderson-Darling teststatistikk: {result.statistic:.4f}")
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            print(f"Signifikansnivå {sl}%: kritisk verdi = {cv:.4f}")
            if result.statistic > cv:
                print("❗ Avviker fra normalfordeling")
            else:
                print("✅ Kan anta normalfordeling")
    """

    # Uavhengig t-test (for lik varians: equal_var=True)
    t_stat, p_value_two_tailed = ttest_ind(group3, group1, equal_var=True)

    # Gjør testen ensidig (H₁: μ₃ < μ₁)
    p_value_one_tailed = p_value_two_tailed / 2

    # Signifikansnivå
    alpha = 0.05

    # Utskrift av resultatene
    print("=== Hypothesis Test: μ₃ < μ₁ ===")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value (one-tailed): {p_value_one_tailed:.4f}")

    # Beslutte om vi skal forkaste nullhypotesen
    if (t_stat < 0) and (p_value_one_tailed < alpha):
        print("❗ We reject the null hypothesis: Group 3 has significantly lower mean distance than group 1.")
    else:
        print("✅ We fail to reject the null hypothesis: No significant difference in mean distance between group 3 and group 1.")
    
    stat, p_value = f_oneway(group1, group2, group3)
    print(f"ANOVA F-statistic: {stat}")
    print(f"ANOVA P-value: {p_value}")


def plotting():
    # === Visualiseringer ===
    # --- 1. Konfidensintervall for gjennomsnitt med feilstolper ---
    plt.figure(figsize=(10, 6))
    for idx, group_data in enumerate([group1, group2, group3], start=1):
        mean = group_data.mean()
        ci_lower = group_data.mean() - (1.96 * group_data.std() / np.sqrt(len(group_data)))
        ci_upper = group_data.mean() + (1.96 * group_data.std() / np.sqrt(len(group_data)))
        
        # Juster plasseringen til gruppen nærmere midten
        plt.errorbar(
            x=[idx],  # Sette gruppen litt nærmere midten
            y=[mean],
            yerr=[[mean - ci_lower], [ci_upper - mean]],  # Feilmarginen (konfidensintervall)
            fmt='o',  # Bruker sirkelpunkt for gjennomsnitt
            color=['blue', 'green', 'red'][idx-1],  # Farge for hver gruppe
            capsize=5,  # Legger til små cap'er på errorbarene
            elinewidth=2,  # Øker tykkelsen på errorbarene
            markersize=8,  # Størrelse på markørene
            label=f"Group {idx}"  # Legger til etikett for gruppene
        )

    # Legge til stil for diagrammet
    plt.title("Mean Distance with 95% Confidence Intervals")
    #plt.xlabel("Groups")
    plt.ylabel("Mean Distance")
    plt.xticks([1, 2, 3], ['Group 1', 'Group 2', 'Group 3'])  # Setter etikett for x-aksen
    plt.grid(axis='y')  # Legger til horisontale gridlines, fjerner vertikale
    plt.legend()

    plt.tight_layout()
    plt.savefig("ConfidenceInterval_meanDistance.png", dpi=300)
    plt.show()

    # --- 2. Histogrammer med KDE ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 rad og 3 kolonner

    # For Gruppe 1
    sns.histplot(group1, kde=True, ax=axes[0], color="blue")
    axes[0].set_title("Group 1: Histogram & KDE")
    axes[0].set_xlabel("Mean Distance")
    axes[0].set_ylabel("Density")

    # For Gruppe 2
    sns.histplot(group2, kde=True, ax=axes[1], color="green")
    axes[1].set_title("Group 2: Histogram & KDE")
    axes[1].set_xlabel("Mean Distance")
    axes[1].set_ylabel("Density")

    # For Gruppe 3
    sns.histplot(group3, kde=True, ax=axes[2], color="red")
    axes[2].set_title("Group 3: Histogram & KDE")
    axes[2].set_xlabel("Mean Distance")
    axes[2].set_ylabel("Density")

    # Justere layout for bedre plass
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

def confidence_interval_median(data, confidence=0.95):
    n = len(data)
    median = data.median()
    std = data.std(ddof=1)
    t_crit = t.ppf((1 + confidence) / 2., n - 1)
    margin_of_error = t_crit * std / np.sqrt(n)
    lower = median - margin_of_error
    upper = median + margin_of_error
    return median, std, n, lower, upper



def StatisticalHypothesisTestMedian():
    # === Steg 1 & 2: Beregn estimater og CI for hver gruppe ===
    results = []
    for group in sorted(summary_df["Group Number"].unique()):
        group_data = summary_df[summary_df["Group Number"] == group]["Median Distance"].dropna()
        
        median, std, n, ci_lower, ci_upper = confidence_interval_median(group_data)
        
        results.append({
            "Group": group,
            "N": n,
            "Median Distance": median,
            "Standard Deviation": std,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper
        })

    # === Lag en ny DataFrame med resultatene ===
    results_df = pd.DataFrame(results)

    # === Vis og/eller lagre resultatene ===
    print(results_df.to_string(index=False))

    # Du kan lagre det til CSV hvis ønskelig
    results_df.to_csv("group_distance_CI_summary_median.csv", index=False)

     # === Ekstra: Test likhet i varians mellom gruppe 1 og gruppe 3 ===
    group1 = summary_df[summary_df["Group Number"] == 1]["Median Distance"].dropna()
    group2 = summary_df[summary_df["Group Number"] == 2]["Median Distance"].dropna()
    group3 = summary_df[summary_df["Group Number"] == 3]["Median Distance"].dropna()

    # Levene's test (robust for ulik fordeling)
    stat_lev, p_lev = levene(group1, group3)
    print(f"\nLevene's test for lik varians: p = {p_lev:.4f}")

    # Bartlett's test (forutsetter normalfordeling)
    stat_bart, p_bart = bartlett(group1, group3)
    print(f"Bartlett's test for lik varians: p = {p_bart:.4f}")

    # Tolkning
    if p_lev < 0.05 or p_bart < 0.05:
        print("❗ Det er signifikant forskjell i varians mellom gruppe 1 og 3.")
    else:
        print("✅ Ingen signifikant forskjell i varians mellom gruppe 1 og 3.")


    # === Normalitetstester: Shapiro-Wilk ===
    print("\n=== Test av normalfordeling for hver gruppe (Shapiro-Wilk) ===")
    for group in sorted(summary_df["Group Number"].unique()):
        group_data = summary_df[summary_df["Group Number"] == group]["Median Distance"].dropna()
        stat, p = shapiro(group_data)
        print(f"Gruppe {group}: p = {p:.4f}", end=" → ")
        if p < 0.05:
            print("❗ Avviker signifikant fra normalfordeling")
        else:
            print("✅ Kan anta normalfordeling")

    # Uavhengig t-test (for lik varians: equal_var=True)
    t_stat, p_value_two_tailed = ttest_ind(group3, group1, equal_var=True)

    # Gjør testen ensidig (H₁: μ₃ < μ₁)
    p_value_one_tailed = p_value_two_tailed / 2

    # Signifikansnivå
    alpha = 0.05

    # Utskrift av resultatene
    print("=== Hypothesis Test: μ₃ < μ₁ ===")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value (one-tailed): {p_value_one_tailed:.4f}")

    # Beslutte om vi skal forkaste nullhypotesen
    if (t_stat < 0) and (p_value_one_tailed < alpha):
        print("❗ We reject the null hypothesis: Group 3 has significantly lower median distance than group 1.")
    else:
        print("✅ We fail to reject the null hypothesis: No significant difference in median distance between group 3 and group 1.")
    
    stat, p_value = f_oneway(group1, group2, group3)
    print(f"ANOVA F-statistic: {stat}")
    print(f"ANOVA P-value: {p_value}")


def confidence_interval_min(data, confidence=0.95):
    n = len(data)
    min = data.min()
    std = data.std(ddof=1)
    t_crit = t.ppf((1 + confidence) / 2., n - 1)
    margin_of_error = t_crit * std / np.sqrt(n)
    lower = min - margin_of_error
    upper = min + margin_of_error
    return min, std, n, lower, upper

def plotting_min(group1, group2, group3):
    # === Visualiseringer ===
    # --- 1. Konfidensintervall for gjennomsnitt med feilstolper ---
    plt.figure(figsize=(10, 6))
    for idx, group_data in enumerate([group1, group2, group3], start=1):
        min = group_data.min()
        ci_lower = group_data.min() - (1.96 * group_data.std() / np.sqrt(len(group_data)))
        ci_upper = group_data.min() + (1.96 * group_data.std() / np.sqrt(len(group_data)))
        
        # Juster plasseringen til gruppen nærmere midten
        plt.errorbar(
            x=[idx],  # Sette gruppen litt nærmere midten
            y=[min],
            yerr=[[min - ci_lower], [ci_upper - min]],  # Feilmarginen (konfidensintervall)
            fmt='o',  # Bruker sirkelpunkt for gjennomsnitt
            color=['blue', 'green', 'red'][idx-1],  # Farge for hver gruppe
            capsize=5,  # Legger til små cap'er på errorbarene
            elinewidth=2,  # Øker tykkelsen på errorbarene
            markersize=8,  # Størrelse på markørene
            label=f"Group {idx}"  # Legger til etikett for gruppene
        )

    # Legge til stil for diagrammet
    plt.title("Min Distance with 95% Confidence Intervals")
    #plt.xlabel("Groups")
    plt.ylabel("Min Distance")
    plt.xticks([1, 2, 3], ['Group 1', 'Group 2', 'Group 3'])  # Setter etikett for x-aksen
    plt.grid(axis='y')  # Legger til horisontale gridlines, fjerner vertikale
    plt.legend()

    plt.tight_layout()
    plt.savefig("ConfidenceInterval_minDistance.png", dpi=300)
    plt.show()

    # --- 2. Histogrammer med KDE ---
    fig, axes = plt.subplots(3, 1, figsize=(6, 10))  # 1 rad og 3 kolonner

    # For Gruppe 1
    sns.histplot(group1, kde=True, ax=axes[0], color="blue")
    axes[0].set_title("Group 1: Histogram & KDE")
    axes[0].set_xlabel("Min Distance")
    axes[0].set_ylabel("Density")

    # For Gruppe 2
    sns.histplot(group2, kde=True, ax=axes[1], color="green")
    axes[1].set_title("Group 2: Histogram & KDE")
    axes[1].set_xlabel("Min Distance")
    axes[1].set_ylabel("Density")

    # For Gruppe 3
    sns.histplot(group3, kde=True, ax=axes[2], color="red")
    axes[2].set_title("Group 3: Histogram & KDE")
    axes[2].set_xlabel("Min Distance")
    axes[2].set_ylabel("Density")

    # Justere layout for bedre plass
    plt.tight_layout()
    plt.savefig("Histogram_KDE_min.png", dpi=300)
    plt.show()

    # --- 3. QQ-plots ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    for ax, data, label in zip(axes, [group1, group2, group3], ["Group 1", "Group 2", "Group 3"]):
        sm.qqplot(data, line="s", ax=ax)
        ax.set_title(f"QQ-plot: {label}")
    plt.tight_layout()
    plt.savefig("QQplots_min.png", dpi=300)
    plt.show()

    # --- 4. Boxplot ---
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="Group Number", y="Min Distance", data=summary_df, hue="Group Number", palette="Set2", legend=False)
    plt.title("Boxplot of Min Distance per Group")
    plt.xlabel("Group")
    plt.ylabel("Min Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("boxplot_min_distance.png", dpi=300)
    plt.show()


def StatisticalHypothesisTestmin():
    # === Steg 1 & 2: Beregn estimater og CI for hver gruppe ===
    results = []

    for group in sorted(summary_df["Group Number"].unique()):
        group_data = summary_df[summary_df["Group Number"] == group]["Min Distance"].dropna()
        
        min_d, std, n, ci_lower, ci_upper = confidence_interval_min(group_data)
        
        results.append({
            "Group": group,
            "N": n,
            "Min Distance": min_d,
            "Standard Deviation": std,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper
        })

    # === Lag en ny DataFrame med resultatene ===
    results_df = pd.DataFrame(results)

    # === Vis og/eller lagre resultatene ===
    print(results_df.to_string(index=False))

    # Du kan lagre det til CSV hvis ønskelig
    results_df.to_csv("group_distance_CI_summary_median.csv", index=False)

     # === Ekstra: Test likhet i varians mellom gruppe 1 og gruppe 3 ===
    group1 = summary_df[summary_df["Group Number"] == 1]["Min Distance"].dropna()
    group2 = summary_df[summary_df["Group Number"] == 2]["Min Distance"].dropna()
    group3 = summary_df[summary_df["Group Number"] == 3]["Min Distance"].dropna()

    # Levene's test (robust for ulik fordeling)
    stat_lev, p_lev = levene(group1, group3)
    print(f"\nLevene's test for lik varians: p = {p_lev:.4f}")

    # Bartlett's test (forutsetter normalfordeling)
    stat_bart_3, p_bart_3 = bartlett(group2, group3)
    print(f"Bartlett's test for lik varians (2 og 3): p = {p_bart_3:.4f}")
    # Bartlett's test (forutsetter normalfordeling)
    stat_bart_2, p_bart_2 = bartlett(group1, group2)
    print(f"Bartlett's test for lik varians (1 og 2): p = {p_bart_2:.4f}")
    # Bartlett's test (forutsetter normalfordeling)
    stat_bart, p_bart = bartlett(group1, group3)
    print(f"Bartlett's test for lik varians (1 og 3): p = {p_bart:.4f}")

    # Tolkning
    if p_lev < 0.05 or p_bart < 0.05:
        print("❗ Det er signifikant forskjell i varians mellom gruppe 1 og 3.")
    else:
        print("✅ Ingen signifikant forskjell i varians mellom gruppe 1 og 3.")


    # === Normalitetstester: Shapiro-Wilk ===
    print("\n=== Test av normalfordeling for hver gruppe (Shapiro-Wilk) ===")
    for group in sorted(summary_df["Group Number"].unique()):
        group_data = summary_df[summary_df["Group Number"] == group]["Min Distance"].dropna()
        stat, p = shapiro(group_data)
        print(f"Gruppe {group}: p = {p:.4f}", end=" → ")
        if p < 0.05:
            print("❗ Avviker signifikant fra normalfordeling")
        else:
            print("✅ Kan anta normalfordeling")

    #Uavhengig t-test (for lik varians: equal_var=True)
    t_stat, p_value_two_tailed = ttest_ind(group1, group3, equal_var=True)
    # Gjør testen ensidig (H₁: μ₃ < μ₁)
    p_value_one_tailed = p_value_two_tailed / 2

    # Signifikansnivå
    alpha = 0.05

    print("=== Hypothesis Test: μ₁ < μ₃ ===")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value (one-tailed): {p_value_one_tailed:.4f}")

    if (t_stat < 0) and (p_value_one_tailed < alpha):
        print("❗ We reject the null hypothesis: Group 1 is significantly closer to the protest than Group 3.")
    else:
        print("✅ We fail to reject the null hypothesis: No significant difference.")
    
    stat, p_value = f_oneway(group1, group2, group3)
    print(f"ANOVA F-statistic: {stat}")
    print(f"ANOVA P-value: {p_value}")

    plotting_min(group1, group2, group3)



def test_normality():
    # === Normalitetstester: Shapiro-Wilk ===
    print("\n=== Test av normalfordeling for hver gruppe (Shapiro-Wilk) ===")
    for group in sorted(summary_df["Group Number"].unique()):
        group_data = summary_df[summary_df["Group Number"] == group]["Median Distance"].dropna()
        stat, p = shapiro(group_data)
        print(f"Gruppe {group}: p = {p:.4f}", end=" → ")
        if p < 0.05:
            print("❗ Avviker signifikant fra normalfordeling")
        else:
            print("✅ Kan anta normalfordeling")

#test_normality()

StatisticalHypothesisTestmin()