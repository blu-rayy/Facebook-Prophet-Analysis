import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("data/Annual 2023 Profits.csv")  # Ensure correct file path

# Drop any unnamed columns
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors='ignore')

# Define month mapping
month_mapping = {
    "January": "2023-01-01", "February": "2023-02-01", "March": "2023-03-01",
    "April": "2023-04-01", "May": "2023-05-01", "June": "2023-06-01",
    "July": "2023-07-01", "August": "2023-08-01", "September": "2023-09-01",
    "October": "2023-10-01", "November": "2023-11-01", "December": "2023-12-01"
}

# Convert 'Month' column using the mapping
df['ds'] = df['Month'].map(month_mapping)
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

# ====== 1. COMBINED FORECAST (Total Profit) ====== #
df_total = df.groupby(['ds'], as_index=False).agg({'Total Profit': 'sum'})

# Rename columns for Prophet
df_total = df_total.rename(columns={'Total Profit': 'y'})

# Fit Prophet model with multiplicative seasonality
model_total = Prophet(seasonality_mode='multiplicative')
model_total.fit(df_total[['ds', 'y']])

# Create future dataframe (1-year prediction)
future_total = model_total.make_future_dataframe(periods=12, freq='M')
forecast_total = model_total.predict(future_total)

# Calculate evaluation metrics
actual_values = df_total['y'].values
predicted_values = forecast_total['yhat'][:len(df_total)].values

mae = mean_absolute_error(actual_values, predicted_values)
mse = mean_squared_error(actual_values, predicted_values)
rmse = mse ** 0.5
mape = (abs(actual_values - predicted_values) / actual_values).mean() * 100

# Print evaluation results
print("\n=== Evaluation Metrics for Combined Forecast ===")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plot combined forecast
plt.figure(figsize=(10, 5))
plt.plot(df_total['ds'], df_total['y'], label="Actual Total Profit", marker='o')
plt.plot(forecast_total['ds'], forecast_total['yhat'], linestyle='dashed', label="Forecasted Total Profit")
plt.fill_between(forecast_total['ds'], forecast_total['yhat_lower'], forecast_total['yhat_upper'], alpha=0.2)
plt.title("Combined Profit Forecast (All Categories)")
plt.xlabel("Date")
plt.ylabel("Total Profit")
plt.legend()
plt.grid()
plt.show()

# ====== 2. INDIVIDUAL FORECASTS (Per Category) ====== #
df_grouped = df.groupby(['ds', 'Category'], as_index=False).agg({'Total Profit': 'sum'})
categories = df_grouped['Category'].unique()

for category in categories:
    category_df = df_grouped[df_grouped['Category'] == category]

    # Ensure at least 2 data points
    if len(category_df) < 2:
        print(f"Skipping {category}: Not enough data")
        continue

    # Rename columns for Prophet
    category_df = category_df.rename(columns={'Total Profit': 'y'})

    # Fit Prophet model with multiplicative seasonality
    model_category = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.1)
    model_category.fit(category_df[['ds', 'y']])

    # Create future dataframe (1-year prediction)
    future_category = model_category.make_future_dataframe(periods=12, freq='M')
    forecast_category = model_category.predict(future_category)


    # Calculate evaluation metrics
    actual_values = category_df['y'].values
    predicted_values = forecast_category['yhat'][:len(category_df)].values

    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = mse ** 0.5
    mape = (abs(actual_values - predicted_values) / actual_values).mean() * 100

    # Print evaluation results
    print(f"\n=== Evaluation Metrics for {category} ===")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # Plot individual forecast
    plt.figure(figsize=(10, 5))
    plt.plot(category_df['ds'], category_df['y'], label="Actual Profit", marker='o')
    plt.plot(forecast_category['ds'], forecast_category['yhat'], linestyle='dashed', label="Forecasted Profit")
    plt.fill_between(forecast_category['ds'], forecast_category['yhat_lower'], forecast_category['yhat_upper'], alpha=0.2)
    plt.title(f"Profit Forecast for {category}")
    plt.xlabel("Date")
    plt.ylabel("Total Profit")
    plt.legend()
    plt.grid()
    plt.show()