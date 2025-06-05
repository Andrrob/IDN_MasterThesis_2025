import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.stats import t, levene, bartlett, shapiro, ttest_ind, mannwhitneyu, f_oneway, kruskal

# This script performs data preprocessing, statistical tests (normality, variance, group differences),
# and multiple linear regression analyses on the dataset, then exports the regression results to Excel.

df = pd.read_csv(file_path)

# Performs the Shapiro-Wilk test for normality
def Shapiro_Wilk_test(data):
    stat, p = shapiro(data)
            #print(f"Gruppe {group}: p = {p:.4f}", end=" → ")
    if p < 0.05:
        print("❗ Deviates significantly from normal distribution")
        return p, False
    else:
        print("✅ ssumption of normality is acceptable")
        return p, True

# Performs a one-way ANOVA test to determine if there are significant differences between three groups assuming normal distribution and equal variances.
def ANOVA_h_test(d1, d2, d3):
    f_stat, p_value = f_oneway(d1, d2, d3)
    alpha = 0.05
    print(f"P-value: {p_value}, F-stat: {f_stat}")
    if p_value < alpha:
        print("There is a statistically significant difference between the groups.")
    else:
        print("No statistically significant difference was found between the groups.")


# Performs the Kruskal-Wallis test (non-parametric alternative to ANOVA) to check if the three groups differ significantly.
def kursal_h_test(d1, d2, d3):
    test_stat, p_value = kruskal(d1, d2, d3)
    alpha = 0.05
    print(f"P-value: {p_value}, Test-stat: {test_stat}")
    if p_value < alpha:
        print("There is a statistically significant difference between the groups.")
    else:
        print("No statistically significant difference was found between the groups.")

# Analyzes the "Movement" variable across three groups by checking for normality (Shapiro-Wilk), homogeneity of variance (Bartlett and Levene), and group differences using
def analyse_movment():
    group1 = df[df["Group Number"] == 1]["Movment"].dropna()
    group2 = df[df["Group Number"] == 2]["Movment"].dropna()
    group3 = df[df["Group Number"] == 3]["Movment"].dropna()

    print(f"Movment all: {df["Movment"].mean()}")
    print(f"Movment Group 1: {group1.mean()}")
    print(f"Movment Group 2: {group2.mean()}")
    print(f"Movment Group 3: {group3.mean()}")

    p1, _ = Shapiro_Wilk_test(group1)
    p2, _ = Shapiro_Wilk_test(group2)
    p3, _ =Shapiro_Wilk_test(group3)
    print(f"P-values from Shapiro_Wilk: Group 1: {p1}, Group 2: {p2}, Group 3: {p3}")
    _, bartlett_p = bartlett(group1, group2, group3)
    if bartlett_p > 0.05:
        print(f"P-verdi: {bartlett_p}, good")
    else:
        print(f"P-verdi: {bartlett_p}, not good")
    _, levene_p = levene(group1, group2, group3)
    if levene_p > 0.05:
        print(f"P-verdi: {levene_p}, good")
    else:
        print(f"P-verdi: {levene_p}, not good")

    kursal_h_test(group1, group2, group3)


# Analyzes the "Immersiveness" variable across three groups. Tests for normality (Shapiro-Wilk), equal variances (Bartlett and Levene), and uses ANOVA to assess group differences if assumptions are met.
def analyse_immersivness():
    group1 = df[df["Group Number"] == 1]["Immersiveness"].dropna()
    group2 = df[df["Group Number"] == 2]["Immersiveness"].dropna()
    group3 = df[df["Group Number"] == 3]["Immersiveness"].dropna()
    p1, _ = Shapiro_Wilk_test(group1)
    p2, _ = Shapiro_Wilk_test(group2)
    p3, _ = Shapiro_Wilk_test(group3)
    print(f"P-values from Shapiro_Wilk: Group 1: {p1}, Group 2: {p2}, Group 3: {p3}")
    _, bartlett_p = bartlett(group1, group2, group3)
    if bartlett_p > 0.05:
        print(f"P-verdi: {bartlett_p}, good")
    else:
        print(f"P-verdi: {bartlett_p}, not good")
    _, levene_p = levene(group1, group2, group3)
    if levene_p > 0.05:
        print(f"P-verdi: {levene_p}, good")
    else:
        print(f"P-verdi: {levene_p}, not good")
    ANOVA_h_test(group1, group2, group3)

#analyse_immersivness()







# Select independent variables (X) and dependent variable (Y)
X_other = df[['Gender', 'Gaming', 'Protest', 'VR-headset', 'Q14', 'Awareness of AI', 'Immersiveness','Movment', 'mean_hrv']]
X = pd.concat([X_other, X_goal], axis=1)
X = sm.add_constant(X) 
X = X.astype(float)
#print(X.dtypes)

#print(type(X))
if X.isna().sum().sum() > 0:
    print("NaN values found in X")
    X = X.dropna()

# Performs a multiple linear regression with "Min Distance" as the dependent variable.
def regresjon_Dis():
    y1 = df['Min Distance']
    model1 = sm.OLS(y1, X).fit()
    print("Regresjon for Min Distance:\n")
    print(model1.summary())

    coef_table = model1.summary2().tables[1]  # Get the summary table as a DataFrame
    # Extract key statistics from the model
    header_stats = {
        "R-squared": model1.rsquared,
        "Adj. R-squared": model1.rsquared_adj,
        "F-statistic": model1.fvalue,
        "Prob (F-statistic)": model1.f_pvalue,
        "Log-Likelihood": model1.llf,
        "AIC": model1.aic,
        "BIC": model1.bic,
        "No. Observations": model1.nobs,
        "Df Model": model1.df_model,
        "Df Residuals": model1.df_resid
    }
    header_df = pd.DataFrame(header_stats.items(), columns=["Statistic", "Value"])
    
    with pd.ExcelWriter("regression_results_mindis.xlsx", engine='xlsxwriter') as writer:
        header_df.to_excel(writer, sheet_name='Min Distance Regression', index=False, startrow=0)
        coef_table.to_excel(writer, sheet_name='Min Distance Regression', startrow=len(header_df) + 3)

        # Add formatting for clarity
        workbook = writer.book
        worksheet = writer.sheets['Min Distance Regression']

        # Bold headers
        bold_format = workbook.add_format({'bold': True})
        worksheet.write(0, 0, "Statistic", bold_format)
        worksheet.write(0, 1, "Value", bold_format)
        worksheet.write(len(header_df) + 3, 0, coef_table.columns.name or "", bold_format)

        # Add section titles
        worksheet.write(len(header_df) + 2, 0, "Regression Coefficients", bold_format)

# Performs a multiple linear regression with "mean HRV" (Heart Rate Variability) as the dependent variable.
def regresjon_HRV():
    y2 = df['mean_hrv']
    model2 = sm.OLS(y2, X).fit()
    print("\nRegresjon for HRV:\n")
    print(model2.summary())

    # Get coefficient table
    coef_table = model2.summary2().tables[1]

    # Extract key model statistics
    header_stats = {
        "R-squared": model2.rsquared,
        "Adjusted R-squared": model2.rsquared_adj,
        "F-statistic": model2.fvalue,
        "Prob (F-statistic)": model2.f_pvalue,
        "Log-Likelihood": model2.llf,
        "AIC": model2.aic,
        "BIC": model2.bic,
        "No. Observations": model2.nobs,
        "Df Model": model2.df_model,
        "Df Residuals": model2.df_resid
    }
    header_df = pd.DataFrame(header_stats.items(), columns=["Statistic", "Value"])
                                                            # Write to Excel with formatting
    with pd.ExcelWriter("regression_results_hrv.xlsx", engine='xlsxwriter') as writer:
        header_df.to_excel(writer, sheet_name='HRV_Regression', index=False, startrow=0)
        coef_table.to_excel(writer, sheet_name='HRV_Regression', startrow=len(header_df) + 3)

        # Add formatting for clarity
        workbook = writer.book
        worksheet = writer.sheets['HRV_Regression']

        # Bold headers
        bold_format = workbook.add_format({'bold': True})
        worksheet.write(0, 0, "Statistic", bold_format)
        worksheet.write(0, 1, "Value", bold_format)
        worksheet.write(len(header_df) + 3, 0, coef_table.columns.name or "", bold_format)

        # Add section titles
        worksheet.write(len(header_df) + 2, 0, "Regression Coefficients", bold_format)






   
