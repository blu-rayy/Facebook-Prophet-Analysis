import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from matplotlib.ticker import FuncFormatter

# Load dataset
file_path = r"data\Annual 2023 Profits.csv"  # Adjust as needed
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Debugging: Check column names
print("Columns in dataset:", df.columns)

# Month mapping
month_mapping = {
    "January": "2023-01-01", "February": "2023-02-01", "March": "2023-03-01",
    "April": "2023-04-01", "May": "2023-05-01", "June": "2023-06-01",
    "July": "2023-07-01", "August": "2023-08-01", "September": "2023-09-01",
    "October": "2023-10-01", "November": "2023-11-01", "December": "2023-12-01"
}

# Convert 'Month' column
df['ds'] = df['Month'].map(month_mapping)
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

# Debugging: Check date conversion
print(df[['Month', 'ds']].drop_duplicates().head())

# Group by date and category
df_grouped = df.groupby(['ds', 'Category'], as_index=False).agg({'Total Profit': 'sum'})

# Ensure unique values
df_grouped = df_grouped.drop_duplicates(subset=['ds', 'Category'])

# Define categories to analyze
services = ["Services", "Ultrasound", "Vaccines", "Laboratory", "Medicines"]

# Define plot colors
colors = {
    "Services": "blue", "Ultrasound": "green", "Vaccines": "red",
    "Laboratory": "purple", "Medicines": "orange"
}

plt.figure(figsize=(14, 7))

# Create a dictionary to store future predictions
future_profits = {}

for category in services:
    category_df = df_grouped[df_grouped["Category"] == category].sort_values(by='ds')

    if len(category_df) < 2:
        print(f"Skipping {category}: Not enough data")
        continue

    category_df = category_df.rename(columns={'Total Profit': 'y'})

    # Train Prophet model
    model = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.5)
    model.fit(category_df[['ds', 'y']])

    # Forecast future profits
    future = model.make_future_dataframe(periods=12, freq='ME')
    forecast = model.predict(future)

    # Store predictions for the last 12 months
    future_profits[category] = forecast[['ds', 'yhat']].tail(12)

    # Plot actual values
    plt.plot(category_df['ds'], category_df['y'], marker='o', label=f"Actual {category}", color=colors[category])

    # Plot predictions
    plt.plot(forecast['ds'], forecast['yhat'], linestyle='dashed', label=f"Predicted {category}", color=colors[category])

# Format y-axis for readability
def format_profit(value, _):
    if value >= 1_000_000:
        return f'{value/1_000_000:.1f}M'
    elif value >= 1_000:
        return f'{value/1_000:.0f}K'
    return str(value)

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_profit))

plt.title("Profit Forecast for Top 5 Categories")
plt.xlabel("Date")
plt.ylabel("Total Profit")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
plt.grid()
plt.tight_layout()
plt.show()

# Display the future profits table
print("\n📊 **Future Profit Forecast for Next 12 Months:**")
for category, data in future_profits.items():
    print(f"\n🔹 {category} Profit Forecast:")
    print(data.to_string(index=False, formatters={'yhat': lambda x: f"{x:,.2f}"}))
