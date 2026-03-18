import pandas as pd
import os

base_path = os.path.dirname(__file__)  # SCRIPTS folder
data_path = os.path.join(base_path, "..", "DATA")

input_file = os.path.join(data_path, "Combined_Dataset.csv")
output_file = os.path.join(data_path, "cleaned_dataset.csv")

df = pd.read_csv(input_file)

df["Age-adjusted Death Rate"] = (
    df["Age-adjusted Death Rate"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

df["Group"] = df["Race"] + "_" + df["Sex"]
df["Group"] = df["Group"].str.replace(" ", "")

pivot_df = df.pivot_table(
    index="Year",
    columns="Group",
    values=[
        "Average Life Expectancy (Years)",
        "Age-adjusted Death Rate"
    ]
)

pivot_df.columns = [
    f"{metric}_{group}".replace(" ", "_")
    for metric, group in pivot_df.columns
]

pivot_df = pivot_df.reset_index()

spending_cols = [
    "Year",
    "Social Security", "Social Security (GDP)",
    "Medicare", "Medicare (GDP)",
    "Medicaid", "Medicaid (GDP)",
    "Income security", "Income security (GDP)",
    "Major health care programs", "Major health care programs (GDP)",
    "All Families Below Poverty (With and Without Children) Percent",
    "All Families Below Poverty (Without Children) Percent"
]

spending_df = df[spending_cols].drop_duplicates(subset=["Year"])

final_df = pd.merge(spending_df, pivot_df, on="Year")

final_df.columns = final_df.columns.str.replace(" ", "_")

final_df = final_df.sort_values(by="Year")

final_df.to_csv(output_file, index=False)

print("Shape:", final_df.shape)
print(final_df.head())