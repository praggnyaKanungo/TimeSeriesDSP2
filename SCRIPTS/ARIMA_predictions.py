import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings("ignore")

os.makedirs("../OUTPUTS", exist_ok=True)

df = pd.read_csv("../DATA/lag_feature_dataset.csv")
df = df.sort_values("Year")

predictor_cols = [
    "Social_Security_(GDP)_lag1",
    "Medicare_(GDP)_lag1",
    "Medicaid_(GDP)_lag1",
    "Income_security_(GDP)_lag1",
    "Major_health_care_programs_(GDP)_lag1"
]

exclude_cols = ["Year"] + predictor_cols

outcomes = [col for col in df.columns if col not in exclude_cols]

results = []

# -----------------------------
# MODEL EVALUATION
# -----------------------------
for col in outcomes:

    series = df[col].dropna()

    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    try:
        model = ARIMA(train, order=(1,1,1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=len(test))

        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))

        results.append({
            "Outcome": col,
            "MAE": mae,
            "RMSE": rmse
        })

        print(f"\n=== {col} ===")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")

    except Exception as e:
        print(f"Error with {col}: {e}")

results_df = pd.DataFrame(results)
results_df.to_csv("../OUTPUTS/arima_results.csv", index=False)

print("\nFinal Results:")
print(results_df)

future_steps = 10
future_results = []

for col in outcomes:

    series = df[col].dropna()

    try:
        model = ARIMA(series, order=(1,1,1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=future_steps)

        last_year = df["Year"].max()
        future_years = [last_year + i for i in range(1, future_steps + 1)]

        forecast_df = pd.DataFrame({
            "Year": future_years,
            "Outcome": col,
            "Forecast": forecast
        })

        future_results.append(forecast_df)

    except Exception as e:
        print(f"Error forecasting {col}: {e}")

future_df = pd.concat(future_results)

future_df.to_csv("../OUTPUTS/future_forecasts.csv", index=False)

print(future_df.head())