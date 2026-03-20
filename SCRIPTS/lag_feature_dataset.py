import pandas as pd

# Load data
df = pd.read_csv("../DATA/cleaned_dataset.csv")

# Clean column names
df.columns = df.columns.str.replace(" ", "_")

df = df.sort_values("Year")

predictors = [
    "Social_Security_(GDP)",
    "Medicare_(GDP)",
    "Medicaid_(GDP)",
    "Income_security_(GDP)",
    "Major_health_care_programs_(GDP)"
]

for col in predictors:
    df[col + "_lag1"] = df[col].shift(1)

lag_cols = [col + "_lag1" for col in predictors]
df = df.dropna(subset=lag_cols)

# Save dataset
df.to_csv("../DATA/lag_feature_dataset.csv", index=False)

print("Lag feature dataset created successfully!")