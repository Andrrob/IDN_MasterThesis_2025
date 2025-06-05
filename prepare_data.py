import pandas as pd


# This script cleans and merges survey and experiment data


df_demographics = pd.read_csv(".csv") 
df_data_summary = pd.read_csv(".csv") # Summary file created in AnalyseData.py

df = pd.merge(df_demographics, df_data_summary, on="Experiment Number")


# Gender: Female = 0, Male = 1
df.rename(columns={"Sex": "Gender"}, inplace=True)
df["Gender"] = df["Gender"].str.strip()
df["Gender"] = df["Gender"].map({"Kvinne": 0, "Mann": 1})

# Age: ordinal
df["Age"] = df["Age"].map({
    "18 - 29": 1,
    "30 - 39": 2,
    "40 - 49": 3,
    "50+": 4
})

# Gaming experience: ordinal
df["Gaming"] = df["Gaming"].str.strip()
df["Gaming"] = df["Gaming"].map({
    "Nei": 0,
    "Litt": 1,
    "Ja": 2
})

# VR headset: ordinal
df.rename(columns={"VR-briller": "VR-headset"}, inplace=True)
df["VR-headset"] = df["VR-headset"].str.strip()
df["VR-headset"] = df["VR-headset"].map({
    "Nei": 0,
    "Litt": 1,
    "Ja": 2
})

# Demonstration: ordinal
df.rename(columns={"Demonstrasjon": "Protest"}, inplace=True)
df["Protest"] = df["Protest"].str.strip()
df["Protest"] = df["Protest"].map({
    "Nei": 0,
    "Litt": 1,
    "Ja": 2
})

# SQ1–SQ7: Already scaled 1–7, ensure numeric
for col in ["SQ1", "SQ2", "SQ3", "SQ4", "SQ5", "SQ6", "SQ7"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# SQ8: Yes = 1, No = 0, Unsure = 2
df.rename(columns={"SQ8": "Q14"}, inplace=True)
df["Q14"] = df["Q14"].str.strip().map({
    "Ja": 1,
    "Nei": 0,
    "Usikker": 2
})

# SQ9: Yes = 1, No = 0
df.rename(columns={"SQ9": "Q15"}, inplace=True)
df["Q15"] = df["Q15"].str.strip().map({
    "Ja": 1,
    "Nei": 0
})

# SQ11: Already scaled 1–5, ensure numeric
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

# SQ12: Awareness of AI
df.rename(columns={"SQ12": "Awareness of AI"}, inplace=True)
df["Awareness of AI"] = df["Awareness of AI"].str.strip()
df["Awareness of AI"] = df["Awareness of AI"].map({
    "Nei": 0,
    "Litt": 1,
    "Ja": 2
})


# AI goal: 1 = participate, 2 = move, 3 = scare, 4 = not participate, 0 = other
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

df['Immersiveness'] = df[['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']].mean(axis=1)
df = df.drop(columns=['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7'])

df['Movment'] = df[['SQ10', 'SQ11']].mean(axis=1)
df = df.drop(columns=['SQ10', 'SQ11'])

df.to_csv("cleaned_data.csv", index=False, encoding="utf-8")

