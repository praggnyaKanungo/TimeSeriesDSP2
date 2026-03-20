import pandas as pd
import os

# Paths
base_path = os.path.dirname(__file__)  # SCRIPTS folder
data_path = os.path.join(base_path, "..", "DATA")

input_file = os.path.join(data_path, "Combined_Dataset.csv")
output_file = os.path.join(data_path, "cleaned_dataset.csv")

# Load data
df = pd.read_csv(input_file)

# Clean death rate (remove commas)
df["Age-adjusted Death Rate"] = (
    df["Age-adjusted Death Rate"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Create Race + Sex group
df["Group"] = df["Race"] + "_" + df["Sex"]
df["Group"] = df["Group"].str.replace(" ", "")

# Pivot ONLY health metrics (correct)
pivot_df = df.pivot_table(
    index="Year",
    columns="Group",
    values=[
        "Average Life Expectancy (Years)",
        "Age-adjusted Death Rate"
    ]
)

# Flatten column names
pivot_df.columns = [
    f"{metric}_{group}".replace(" ", "_")
    for metric, group in pivot_df.columns
]

pivot_df = pivot_df.reset_index()

# ✅ INCLUDE AGE-BASED POVERTY COLUMNS HERE
spending_cols = [
    "Year",

    # Spending
    "Social Security", "Social Security (GDP)",
    "Medicare", "Medicare (GDP)",
    "Medicaid", "Medicaid (GDP)",
    "Income security", "Income security (GDP)",
    "Major health care programs", "Major health care programs (GDP)",

    # Poverty (overall)
    "All Families Below Poverty (With and Without Children) Percent",
    "All Families Below Poverty (Without Children) Percent",

    # Age + Sex poverty breakdowns
    "Male Total Below Poverty Percent",
    "Male Under 18 Below Poverty Percent",
    "Male 18 to 64 Below Poverty Percent",
    "Male 64 and Over Below Poverty Percent",
    "Female Total Below Poverty Percent",
    "Female Under 18 Below Poverty Percent",
    "Female 18 to 64 Below Poverty Percent",
    "Female 64 and Over Below Poverty Percent"
]

spending_df = df[spending_cols].drop_duplicates(subset=["Year"])

# Merge with pivoted health data
final_df = pd.merge(spending_df, pivot_df, on="Year")

# Clean column names
final_df.columns = final_df.columns.str.replace(" ", "_")

# Sort by year
final_df = final_df.sort_values(by="Year")

final_df = final_df.dropna()
final_df.to_csv(output_file, index=False)

print("Shape:", final_df.shape)
print(final_df.head())