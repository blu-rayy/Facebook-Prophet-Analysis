import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Load dataset
file_path = r"data\Annual 2023 Profits.csv"
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Convert 'Month' column to datetime
month_mapping = {
    "January": "2023-01-01", "February": "2023-02-01", "March": "2023-03-01",
    "April": "2023-04-01", "May": "2023-05-01", "June": "2023-06-01",
    "July": "2023-07-01", "August": "2023-08-01", "September": "2023-09-01",
    "October": "2023-10-01", "November": "2023-11-01", "December": "2023-12-01"
}
df['ds'] = df['Month'].map(month_mapping)
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

# Filter data for 'Medicine' in the 'Item' column
medicine_df = df[df['Item'] == 'Medicine']

# Identify the top 5 most profitable 'Options' based on 'Total Profit', excluding "Calcium (Calsure Boncare)"
top_5_options = (
    medicine_df.groupby('Option')['Total Profit']
    .sum()
    .drop("Calcium (Calsure Boncare)", errors='ignore')  # Exclude "Calcium (Calsure Boncare)"
    .nlargest(5)
    .index.tolist()
)

# Plot settings
plt.figure(figsize=(14, 7))
colors = ["blue", "green", "red", "purple", "orange"]

# Forecasting loop for each top option
for i, option in enumerate(top_5_options):
    option_df = medicine_df[medicine_df['Option'] == option]

    # Aggregate total profit by month
    profit_grouped = option_df.groupby('ds', as_index=False).agg({'Total Profit': 'sum'})

    # Handle missing months (fill zero if no profit that month)
    all_months = pd.date_range(start='2023-01-01', end='2023-12-01', freq='MS')
    profit_grouped = profit_grouped.set_index('ds').reindex(all_months, fill_value=0).reset_index()
    profit_grouped.columns = ['ds', 'y']

    # Exclude zero values from the plot for better visualization
    non_zero_data = profit_grouped[profit_grouped['y'] > 0]

    # Check if there are at least two full seasonal cycles
    if len(profit_grouped) < 24:
        seasonal = None
        seasonal_periods = None
    else:
        seasonal = 'add'
        seasonal_periods = 12

    # Train Exponential Smoothing Model with both additive and multiplicative seasonality
    try:
        model_add = ExponentialSmoothing(
            profit_grouped['y'],
            trend='add',
            seasonal='add',
            seasonal_periods=seasonal_periods
        )
        fit_model_add = model_add.fit()

        model_mul = ExponentialSmoothing(
            profit_grouped['y'],
            trend='add',
            seasonal='mul',
            seasonal_periods=seasonal_periods
        )
        fit_model_mul = model_mul.fit()

        # Forecast the next 12 months for both models
        forecast_start = profit_grouped['ds'].iloc[-1] + pd.offsets.MonthBegin(1)
        forecast_add = fit_model_add.forecast(12)
        forecast_mul = fit_model_mul.forecast(12)

        forecast_index = pd.date_range(start=forecast_start, periods=12, freq='MS')

        # Calculate R2 values (we'll compare the actual and forecasted values)
        actual_values = profit_grouped['y'].iloc[-12:].values
        r2_add = 1 - (np.sum((actual_values - forecast_add) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2))
        r2_mul = 1 - (np.sum((actual_values - forecast_mul) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2))

        print(f"R2 for Additive Model {option}: {r2_add}")
        print(f"R2 for Multiplicative Model {option}: {r2_mul}")

        # Plot actual and forecasted values, adjusting the forecasted plot
        plt.plot(non_zero_data['ds'], non_zero_data['y'], marker='o', label=f"Actual {option}", color=colors[i])
        plt.plot(forecast_index, forecast_add, linestyle='dashed', label=f"Forecasted Additive {option}", color=colors[i])
        plt.plot(forecast_index, forecast_mul, linestyle='dashed', label=f"Forecasted Multiplicative {option}", color=colors[i])

    except Exception as e:
        print(f"Error with option {option}: {e}")

# Format the graph
plt.title("Total Profit Forecast for Top 5 Medicines (ETS Model)")
plt.xlabel("Date")
plt.ylabel("Total Profit")
plt.xticks(rotation=45)
plt.grid()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout()

plt.show()
