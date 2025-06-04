import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.stats import t, levene, bartlett, shapiro, ttest_ind, mannwhitneyu, f_oneway, kruskal

df_demographics = pd.read_csv("demographics.csv") 
df_data_summary = pd.read_csv("distance_hrv_summary.csv")

df = pd.merge(df_demographics, df_data_summary, on="Experiment Number")

#print(df.head())

# Kjønn: Kvinne = 0, Mann = 1
df.rename(columns={"Sex": "Gender"}, inplace=True)
df["Gender"] = df["Gender"].str.strip()
df["Gender"] = df["Gender"].map({"Kvinne": 0, "Mann": 1})

# Alder: ordinal
df["Age"] = df["Age"].map({
    "18 - 29": 1,
    "30 - 39": 2,
    "40 - 49": 3,
    "50+": 4
})

# Gaming-erfaring: ordinal
df["Gaming"] = df["Gaming"].str.strip()
df["Gaming"] = df["Gaming"].map({
    "Nei": 0,
    "Litt": 1,
    "Ja": 2
})

# VR-briller: ordinal
df.rename(columns={"VR-briller": "VR-headset"}, inplace=True)
df["VR-headset"] = df["VR-headset"].str.strip()
df["VR-headset"] = df["VR-headset"].map({
    "Nei": 0,
    "Litt": 1,
    "Ja": 2
})

# Demonstrasjon: ordinal
df.rename(columns={"Demonstrasjon": "Protest"}, inplace=True)
df["Protest"] = df["Protest"].str.strip()
df["Protest"] = df["Protest"].map({
    "Nei": 0,
    "Litt": 1,
    "Ja": 2
})

# SQ1–SQ7: Disse er allerede skalert 1–7, så sørg for at de er numeriske:
for col in ["SQ1", "SQ2", "SQ3", "SQ4", "SQ5", "SQ6", "SQ7"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# SQ8: Ja = 1, Nei = 0, Usikker = NaN
df.rename(columns={"SQ8": "Q14"}, inplace=True)
df["Q14"] = df["Q14"].str.strip().map({
    "Ja": 1,
    "Nei": 0,
    "Usikker": 2
})

# SQ9: Ja = 1, Nei = 0
df.rename(columns={"SQ9": "Q15"}, inplace=True)
df["Q15"] = df["Q15"].str.strip().map({
    "Ja": 1,
    "Nei": 0
})

# SQ11–SQ12: Disse er allerede skalert 1–5, så sørg for at de er numeriske:
for col in ["SQ10", "SQ11"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df["SQ11"] = df["SQ11"].map({
    5: "en",
    4: "to",
    3: "tre",
    2: "fire",
    1: "fem",
})

df["SQ11"] = df["SQ11"].map({
    "en": 1,
    "to": 2,
    "tre": 3,
    "fire": 4,
    "fem": 5,
})

# SQ12
df.rename(columns={"SQ12": "Awareness of AI"}, inplace=True)
df["Awareness of AI"] = df["Awareness of AI"].str.strip()
df["Awareness of AI"] = df["Awareness of AI"].map({
    "Nei": 0,
    "Litt": 1,
    "Ja": 2
})


# 1 = delta,2 = bevege seg, 3 = skremme, 4 = ikke delta, 0= annet.
df['AI_goal']=[3,1,3,4,1,1,0,1,0,1,1,3,0,1,0,0,1,2,1,1,1,0,0,0,2,0,1,0,2,1,0,2,1,1,2,1,2,2,3,2,0,1,1,4,2,3,3,1,3,0,0,1,0,2,0,0,1,4,0,1,2]
#1,0,0,0,2,0,1,0,2,1,0,2,1,1,2,1,2,2,3,2,
#0,1,1,4,2,3,3,1,3,0,0,1,0,2,0,0,1,4,0,1,2
df['AI_goal'] = df['AI_goal'].astype('category')
X_goal = pd.get_dummies(df['AI_goal'], prefix='AI_goal', drop_first=True)

#answer_counts = df["SQ12"].value_counts().sort_index()

#print(answer_counts)
#print(df.head())
# Group by 'Group Number' and count occurrences of each answer
#grouped_counts = df.groupby("Group Number")["SQ12"].value_counts().unstack(fill_value=0).sort_index()

# Optional: Rename columns to original labels
#grouped_counts.columns = ["Nei", "Litt", "Ja"]

#print(grouped_counts)

df['Immersiveness'] = df[['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']].mean(axis=1)
df = df.drop(columns=['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7'])

df['Movment'] = df[['SQ10', 'SQ11']].mean(axis=1)
df = df.drop(columns=['SQ10', 'SQ11'])


def Shapiro_Wilk_test(data):
    stat, p = shapiro(data)
            #print(f"Gruppe {group}: p = {p:.4f}", end=" → ")
    if p < 0.05:
        print("❗ Avviker signifikant fra normalfordeling")
        return p, False
    else:
        print("✅ Kan anta normalfordeling")
        return p, True

def ANOVA_h_test(d1, d2, d3):
    f_stat, p_value = f_oneway(d1, d2, d3)
    alpha = 0.05
    print(f"P-value: {p_value}, F-stat: {f_stat}")
    if p_value < alpha:
        print("There is a statistically significant difference between the groups.")
    else:
        print("No statistically significant difference was found between the groups.")


def kursal_h_test(d1, d2, d3):
    test_stat, p_value = kruskal(d1, d2, d3)
    alpha = 0.05
    print(f"P-value: {p_value}, Test-stat: {test_stat}")
    if p_value < alpha:
        print("There is a statistically significant difference between the groups.")
    else:
        print("No statistically significant difference was found between the groups.")


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


#analyse_movment()




def analyse_immersivness():
    group1 = df[df["Group Number"] == 1]["Immersiveness"].dropna()
    group2 = df[df["Group Number"] == 2]["Immersiveness"].dropna()
    group3 = df[df["Group Number"] == 3]["Immersiveness"].dropna()
    p1, _ = Shapiro_Wilk_test(group1)
    p2, _ = Shapiro_Wilk_test(group2)
    p3, _ = Shapiro_Wilk_test(group3)
    print(f"P-values from Shapiro_Wilk: Group 1: {p1}, Group 2: {p2}, Group 3: {p3}")
    """""
    _, bartlett_p12 = bartlett(group1, group2)
    if bartlett_p12 > 0.05:
        print("good")
    _, bartlett_p13 = bartlett(group1, group3)
    if bartlett_p13 > 0.05:
        print("good")
    _, bartlett_p23 = bartlett(group2, group3)
    if bartlett_p23 > 0.05:
        print("good")
    """
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


df.to_csv("renset_data.csv", index=False, encoding="utf-8")



# Velg uavhengige variabler (X) og avhengig variabel (Y)
X_other = df[['Gender', 'Gaming', 'Protest', 'VR-headset', 'Q14', 'Awareness of AI', 'Immersiveness','Movment', 'mean_hrv']]
X = pd.concat([X_other, X_goal], axis=1)
#X = df[['Sex', 'Age', 'Gaming', 'VR-briller', 'SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7', 'SQ8', 'SQ9', 'SQ10', 'SQ11', 'SQ12']]  # Uavhengige variabler
#X['Immersiveness'] = X[['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']].mean(axis=1)
#X = X.drop(columns=['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']) # Dropp de originale kolonnene
X = sm.add_constant(X) 
X = X.astype(float)
print(X.dtypes)

#print(type(X))
if X.isna().sum().sum() > 0:
    print("NaN values found in X")
    X = X.dropna()

# 4. Første regresjon: Distance
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

        # Optional: Add formatting for clarity
        workbook = writer.book
        worksheet = writer.sheets['Min Distance Regression']

        # Bold headers
        bold_format = workbook.add_format({'bold': True})
        worksheet.write(0, 0, "Statistic", bold_format)
        worksheet.write(0, 1, "Value", bold_format)
        worksheet.write(len(header_df) + 3, 0, coef_table.columns.name or "", bold_format)

        # Add section titles
        worksheet.write(len(header_df) + 2, 0, "Regression Coefficients", bold_format)



# 5. Andre regresjon: HRV
def regresjon_HRV():
    y2 = df['var_hrv']
    model2 = sm.OLS(y2, X).fit()
    print("\nRegresjon for HRV:\n")
    print(model2.summary())

    coef_table = model2.summary2().tables[1]  # Get the summary table as a DataFrame
    #results2_df.to_csv("regression_results_model2.csv") 

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

    header_df = pd.DataFrame.from_dict(header_stats, orient='index', columns=['Value'])
    
    spacer = pd.DataFrame([[""]] * 2, columns=["Value"])  # 2 empty rows
    output = pd.concat([header_df, spacer, coef_table], axis=0)
    output.to_csv("full_regression_results_hrv.csv")


    # Create a DataFrame of coefficients and standard errors
    #coefficients = model2.params
    #standard_errors = model2.bse
    #results = pd.DataFrame({
    #    'Coefficient': coefficients,
    #    'Standard Error': standard_errors
    #})

    # Save the results to a CSV file
    #results.to_csv("coefficients_and_standard_errors_model_hrv.csv")



def regresjon_HRV2():
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

        # Optional: Add formatting for clarity
        workbook = writer.book
        worksheet = writer.sheets['HRV_Regression']

        # Bold headers
        bold_format = workbook.add_format({'bold': True})
        worksheet.write(0, 0, "Statistic", bold_format)
        worksheet.write(0, 1, "Value", bold_format)
        worksheet.write(len(header_df) + 3, 0, coef_table.columns.name or "", bold_format)

        # Add section titles
        worksheet.write(len(header_df) + 2, 0, "Regression Coefficients", bold_format)




regresjon_Dis()

#regresjon_HRV2()

#def regresjon_Imersivness(): ? 
   