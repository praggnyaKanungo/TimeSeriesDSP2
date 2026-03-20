import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import os

os.makedirs("../OUTPUTS", exist_ok=True)

# Load dataset
df = pd.read_csv("../DATA/lag_feature_dataset.csv")

# Predictors
predictors = [
    "Social_Security_(GDP)_lag1",
    "Medicare_(GDP)_lag1",
    "Medicaid_(GDP)_lag1",
    "Income_security_(GDP)_lag1",
    "Major_health_care_programs_(GDP)_lag1"
]

# Responses
responses = [
    "Average_Life_Expectancy_(Years)_AllRaces_Female",
    "Average_Life_Expectancy_(Years)_AllRaces_Male",
    "Age-adjusted_Death_Rate_AllRaces_Female",
    "Age-adjusted_Death_Rate_AllRaces_Male",
    "All_Families_Below_Poverty_(With_and_Without_Children)_Percent",
    "Male_Total_Below_Poverty_Percent",
    "Male_Under_18_Below_Poverty_Percent",
    "Male_18_to_64_Below_Poverty_Percent",
    "Male_64_and_Over_Below_Poverty_Percent",
    "Female_Total_Below_Poverty_Percent",
    "Female_Under_18_Below_Poverty_Percent",
    "Female_18_to_64_Below_Poverty_Percent",
    "Female_64_and_Over_Below_Poverty_Percent"
]

results = []
scaler = StandardScaler()

for predictor in predictors:
    for response in responses:

        temp_df = df[[predictor, response]].dropna()

        if len(temp_df) < 10:
            continue

        X = temp_df[[predictor]]
        y = temp_df[response]

        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        X_const = sm.add_constant(X)

        model = sm.OLS(y, X_const).fit()
        model_std = sm.OLS(y_scaled, sm.add_constant(X_scaled)).fit()

        coef = model.params[predictor]
        pval = model.pvalues[predictor]

        results.append({
            "Predictor": predictor,
            "Response": response,
            "Coefficient": coef,
            "Standardized_Coefficient": model_std.params[1],
            "P-value": pval,
            "Significant (p<0.05)": pval < 0.05,
            "Direction": "Positive" if coef > 0 else "Negative",
            "R-squared": model.rsquared,
            "N": len(temp_df)
        })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["Significant (p<0.05)", "R-squared"],
    ascending=[False, False]
)

results_df.to_csv("../OUTPUTS/regression_results_expanded.csv", index=False)

print(results_df.head(20))