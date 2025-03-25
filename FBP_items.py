import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from matplotlib.ticker import FuncFormatter

# Load dataset
file_path = r"data\Annual 2023 Profits.csv"  # Adjust path as needed
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

# Clean and convert 'Total Profit' column to numeric
df['Total Profit'] = (
    df['Total Profit']
    .astype(str)  # Ensure all values are strings
    .str.replace(',', '', regex=False)  # Remove commas
    .str.strip()  # Remove leading/trailing whitespace
)
df['Total Profit'] = pd.to_numeric(df['Total Profit'], errors='coerce')  # Convert to numeric, invalid values become NaN

# Drop rows with NaN in 'Total Profit'
df = df.dropna(subset=['Total Profit'])

# Define top 10 most profitable items
top_10_items = [
    "Obstetrics and Gynecology", "Obstetrics and Gynecology Consultation",
    "Pediatrics Consultation", "Medicine", "Flu Vaccine", "Rotateq Vaccine",
    'PCV "Pneumoccoccal" Vaccine', "Medical Packages",
    '6 IN 1 "Hexaxim" Vaccine', "Laser Circumcision"
]

# Filter data for top 10 items
df_filtered = df[df['Item'].isin(top_10_items)]

# Aggregate profits by month
df_grouped = df_filtered.groupby(['ds', 'Item'], as_index=False).agg({'Total Profit': 'sum'})

# Store forecast results
forecast_results = {}

# Plot settings
plt.figure(figsize=(14, 7))
colors = ["blue", "green", "red", "purple", "orange", "pink", "cyan", "brown", "gray", "magenta"]

# Forecasting loop for each item
for i, item in enumerate(top_10_items):
    item_df = df_grouped[df_grouped["Item"] == item].sort_values(by="ds")

    if len(item_df) < 2:
        print(f"Skipping {item}: Not enough data")
        continue

    item_df = item_df.rename(columns={'Total Profit': 'y'})

    # Train Prophet model
    model = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.5)
    model.fit(item_df[['ds', 'y']])

    # Predict next 12 months (ensuring it starts from 2023-01)
    future = model.make_future_dataframe(periods=12, freq='ME', include_history=True)
    forecast = model.predict(future)

    # Store forecast results
    forecast_results[item] = forecast[['ds', 'yhat']].rename(columns={'yhat': 'Predicted Profit'})
    forecast_results[item]['Item'] = item  # Add item name for table

    # Plot actual and predicted values
    plt.plot(item_df['ds'], item_df['y'], marker='o', label=f"Actual {item}", color=colors[i])
    plt.plot(forecast['ds'], forecast['yhat'], linestyle='dashed', label=f"Predicted {item}", color=colors[i])

# 📌 Format X-axis for monthly increments
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=12))  # Show 12 ticks (monthly)

# 📌 Format Y-axis to display K (thousands) or M (millions)
def format_profit(value, _):
    if value >= 1_000_000:
        return f'{value/1_000_000:.1f}M'
    elif value >= 1_000:
        return f'{value/1_000:.0f}K'
    return str(value)

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_profit))

# Graph settings
plt.title("Profit Forecast for Top 10 Most Profitable Items")
plt.xlabel("Date")
plt.ylabel("Total Profit")
plt.grid()
plt.tight_layout()

# 📌 Move legend to the right side
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.show()

# 📊 Combine forecast results into a table (ensure all items appear)
forecast_table = pd.concat(forecast_results.values(), ignore_index=True)
forecast_table = forecast_table[['Item', 'ds', 'Predicted Profit']]

print("\n📊 Forecasted Profits Table:")
print(forecast_table.head(230))  # Show first 30 rows for multiple items
