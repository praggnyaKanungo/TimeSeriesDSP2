import pandas as pd
import os

# Paths
DATA_DIR = "../DATA"
input_file = os.path.join(DATA_DIR, "DeathRatesAndLifeExpectancy.csv")
output_file = os.path.join(DATA_DIR, "health_outcomes_by_year.csv")

# Load dataset
df = pd.read_csv(input_file)

df = df.rename(columns={
    "Average Life Expectancy (Years)": "life_expectancy",
    "Age-adjusted Death Rate": "death_rate"
})

# Create a combined column for race + sex
df["group"] = df["Race"] + "_" + df["Sex"]

# Pivot the dataset (long → wide)
df_wide = df.pivot(
    index="Year",
    columns="group",
    values=["life_expectancy", "death_rate"]
)

# Flatten multi-level column names
df_wide.columns = [
    f"{metric}_{group}".replace(" ", "_")
    for metric, group in df_wide.columns
]

# Reset index so Year becomes a column again
df_wide = df_wide.reset_index()

df_wide = df_wide[df_wide["Year"] >= 1970]

# Save cleaned dataset
df_wide.to_csv(output_file, index=False)

print("Dataset reshaped successfully!")
print("Saved to:", output_file)
print(df_wide.head())