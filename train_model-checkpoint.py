import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import requests
import time

## 1. Load the dataset
file_path = r"D:/PowerDemandPrediction/Power Consumption Data.csv"
df = pd.read_csv(file_path)

## 2. Preprocess the data
# Drop unnecessary columns 
print(df.head())  # Display first few rows
print(df.info())  # Check column names and data types
print("Is 'df' defined?", 'df' in globals())  # Should return True
print(df.columns) # Ensure correct column names
df.columns = df.columns.str.strip()  # Strip any extra spaces from column names
columns_to_remove = ["predicted_consumption"]
df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
print(df.columns) # Verify the remaining columns

## Check for missing values
df = df.dropna()
print(df.head())

## Convert Timestamp to DateTime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df[['Timestamp', 'Real_consumption']].dropna()
df = df.sort_values('Timestamp')

# Add lagged features (e.g., previous 1-hour values)
df['Lag_1h'] = df['Real_consumption'].shift(4)  # 4 steps = 1 hour (15-min intervals)
df['Lag_2h'] = df['Real_consumption'].shift(8)  # 2 hours
df['Lag_24h'] = df['Real_consumption'].shift(96)  # 24 hours
df['Lag_48h'] = df['Real_consumption'].shift(192)
df = df.dropna()  # Drop rows with NaN from shifting
# Now try feature extraction
df["Hour"] = df["Timestamp"].dt.hour
df["Day"] = df["Timestamp"].dt.day
df["Month"] = df["Timestamp"].dt.month
df["Weekday"] = df["Timestamp"].dt.weekday  # 0=Monday, 6=Sunday
df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)
df = df.dropna()
print(df.head())

## Feature Selection (predict power demand using relevant features)
features = ["Hour", "Day", "Month", "Weekday" ,"Lag_1h" , "Lag_2h", "Lag_24h","Lag_48h", "Hour_Sin", "Hour_Cos", "Is_Weekend"]   #You can add more features if available
X = df[features]
y = df["Real_consumption"]

## Split the data (80% training,20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# Print shapes to confirm the split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Preview the first few rows
print("\nFirst 5 rows of X_train:")
print(X_train.head())
print("\nFirst 5 rows of y_train:")
print(y_train.head())
print("\nFirst 5 rows of X_test:")
print(X_test.head())
print("\nFirst 5 rows of y_test:")
print(y_test.head())

## 3. Train vs Test Graph
# Plot train vs test
plt.figure(figsize=(14, 6))
timestamps = df['Timestamp']
train_timestamps = timestamps.loc[y_train.index][::100]
test_timestamps = timestamps.loc[y_test.index][::100]

plt.plot(train_timestamps, y_train[::100], label="Training Data", color="blue", linewidth=1)
plt.plot(test_timestamps, y_test[::100], label="Test Data", color="orange", linewidth=1)

# Add vertical line at the split point
split_point = timestamps.loc[y_train.index[-1]]  # Last timestamp of training data
plt.axvline(x=split_point, color='green', linestyle='--', label='Train/Test Split')

plt.xlabel("Time")
plt.ylabel("Power Demand (kW)")
plt.title("Train vs Test Power Demand")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## 4. Train the XGBoost Model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    reg_lambda=5.0,
    reg_alpha=0.5,
    random_state=42
)
model.fit(X_train, y_train)

## 5. Model Evaluation
# Make predictions
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# Evaluate on test set
print("\nModel Performance:")
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.2f} kW")
print(f"Test RMSE: {rmse:.2f} kW")
print(f"Test R²: {r2:.3f}")

# Evaluate on train set
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

print(f"Train MAE: {train_mae:.2f} kW")
print(f"Train RMSE: {train_rmse:.2f} kW")
print(f"Train R²: {train_r2:.3f}")

## 6.Visualization with seaborn

## 6.1 Power consumption over time
sns.set_style("whitegrid")
# Ensure Timestamp is sorted
df = df.sort_values(by="Timestamp")

# Plot Power Consumption over Time
plt.figure(figsize=(12, 6))
plt.plot(df["Timestamp"], df["Real_consumption"], label="Power Consumption (kW)", color="blue", linewidth=1)

# Formatting the graph
plt.xlabel("Time")
plt.ylabel("Power Consumption (kW)")
plt.title("Power Consumption Over Time")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

## 6.2 Power consumption over one week
week_duration = pd.Timedelta(days=7)
split_point = df['Timestamp'].loc[y_train.index[-1]]  # Last timestamp of training data
start_date = split_point - pd.Timedelta(days=3.5)    # 3.5 days before split
end_date = split_point + pd.Timedelta(days=3.5)      # 3.5 days after split

# Filter data for the week
week_mask = (df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)
df_week = df.loc[week_mask]

# Plot power demand over the week
plt.figure(figsize=(14, 6))
plt.plot(df_week['Timestamp'], df_week['Real_consumption'], label="Power Demand", color="blue", linewidth=1)

# Add vertical line at the split point (if within the week)
if split_point >= start_date and split_point <= end_date:
    plt.axvline(x=split_point, color='green', linestyle='--', label='Train/Test Split')

plt.xlabel("Time")
plt.ylabel("Power Demand (kW)")
plt.title("Power Demand Over a Week")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## 6.3 Hourly Power Consumption Within a Day
plt.figure(figsize=(14, 6))
df['Hour'] = df['Timestamp'].dt.hour  # Extract hour
sns.boxplot(x='Hour', y='Real_consumption', hue='Hour', data=df, palette='coolwarm', legend=False)
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Power Demand (kW)")
plt.title("Hourly Power Consumption Distribution Across All Days")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 6.4 Power consumption for a month
plt.figure(figsize=(14, 6))
# Filter for a specific month (adjust year/month as needed)
month_filter = (df['Timestamp'].dt.year == 2024) & (df['Timestamp'].dt.month == 1)
df_month = df[month_filter].copy()
df_month['Day'] = df_month['Timestamp'].dt.day  # Extract day of month
sns.boxplot(x='Day', y='Real_consumption', hue='Day', data=df_month, palette='viridis', legend=False)
plt.xlabel("Day of Month (1-31)")
plt.ylabel("Power Demand (kW)")
plt.title("Daily Power Consumption Distribution in January 2024")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 6.5 Power consumption for 12 months
plt.figure(figsize=(14, 6))
df['Month'] = df['Timestamp'].dt.month  # Extract month (1-12)
sns.boxplot(x='Month', y='Real_consumption', hue='Month', data=df, palette='viridis', legend=False)
plt.xlabel("Month (1-12)")
plt.ylabel("Power Demand (kW)")
plt.title("Power Demand Distribution Across 12 Months")
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.tight_layout()
plt.show()

## 6.6 Actual vs Predicted
plt.figure(figsize=(14, 6))
test_sample = y_test[-1000:]  # Last 1000 points of test set
pred_sample = y_pred[-1000:]  # Last 1000 predicted points
test_sample_timestamps = df['Timestamp'].loc[y_test.index[-1000:]]  # Use loc for label-based indexing


plt.plot(test_sample_timestamps, test_sample, label="Actual Power Demand", color="blue", linewidth=1)
plt.plot(test_sample_timestamps, pred_sample, label="Predicted Power Demand", color="red", linestyle="--", linewidth=1)
plt.xlabel("Time")
plt.ylabel("Power Demand (kW)")
plt.title("Actual vs Predicted Power Demand (Test Set Sample)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## create dataframe to print actual and predicted values
results_df = pd.DataFrame({
    'Timestamp': test_sample_timestamps.reset_index(drop=True),
    'Actual_Power_Demand_kW': test_sample.reset_index(drop=True),
    'Predicted_Power_Demand_kW': pred_sample
})

# Print the first 20 rows for brevity (adjust as needed)
print("\nActual vs Predicted Power Demand (First 20 Points):")
print(results_df.head(20))

# Optionally, save to CSV for further analysis
results_df.to_csv("actual_vs_predicted_power_demand.csv", index=False)
print("\nData saved to 'actual_vs_predicted_power_demand.csv'")

## Actual vs predicted power demand with error percentage
# Define test_sample, pred_sample, and test_sample_timestamps
test_sample = y_test[-1000:]  # Last 1000 points of test set
pred_sample = y_pred[-1000:]  # Last 1000 predicted points
test_sample_timestamps = df['Timestamp'].tail(1000)  # Last 1000 timestamps

# Calculate absolute and relative errors
absolute_errors = np.abs(test_sample - pred_sample)
relative_errors = (absolute_errors / (np.abs(test_sample) + 1e-10)) * 100  # Percentage error

# Create DataFrame
results_df = pd.DataFrame({
    'Timestamp': test_sample_timestamps.reset_index(drop=True),
    'Actual_Power_Demand_kW': pd.Series(test_sample).reset_index(drop=True),
    'Predicted_Power_Demand_kW': pd.Series(pred_sample).reset_index(drop=True),
    'Absolute_Error_kW': pd.Series(absolute_errors).reset_index(drop=True),
    'Relative_Error_%': pd.Series(relative_errors).reset_index(drop=True)
})

# Print first 20 rows
print("\nActual vs Predicted Power Demand with Errors (First 20 Points):")
print(results_df.head(20))

## 6.7 Forecast Plot
plt.figure(figsize=(14, 6))
plt.plot(df['Timestamp'].loc[y_test.index], y_test, label="Actual Power Demand", color="blue", linewidth=1)
plt.plot(df['Timestamp'].loc[y_test.index], y_pred, label="Predicted Power Demand", color="red", linestyle="--", linewidth=1)
plt.xlabel("Time")
plt.ylabel("Power Demand (kW)")
plt.title("Actual vs Predicted Power Demand (Full Test Set)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Daily Forecast (Iterative)
last_data = X_test.iloc[-1:].copy()
predictions = []
last_timestamp = df['Timestamp'].iloc[-1]
time_steps = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=15), periods=96, freq='15min')

for i in range(96):
    pred = model.predict(last_data)[0]
    predictions.append(pred)
    last_data['Lag_1h'] = pred
    last_data['Lag_2h'] = last_data['Lag_1h']
    last_data['Lag_24h'] = last_data['Lag_2h']
    last_data['Lag_48h'] = last_data['Lag_24h']
    current_hour = last_data['Hour'].values[0] + 0.25
    if current_hour >= 24:
        current_hour -= 24
        last_data['Day'] = last_data['Day'] + 1
        last_data['Weekday'] = (last_data['Weekday'] + 1) % 7
        last_data['Is_Weekend'] = int(last_data['Weekday'].values[0] in [5, 6])
    last_data['Hour'] = current_hour
    last_data['Hour_Sin'] = np.sin(2 * np.pi * current_hour / 24)
    last_data['Hour_Cos'] = np.cos(2 * np.pi * current_hour / 24)
  

plt.figure(figsize=(14, 6))
plt.plot(time_steps, predictions, label="Predicted Power Demand (Next Day)", color="red", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Power Demand (kW)")
plt.title("Forecasted Power Demand for Next 24 Hours")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## Create DataFrame to print predictions(for next 24 hours)
forecast_df = pd.DataFrame({
    'Timestamp': time_steps,
    'Predicted_Power_Demand_kW': predictions
})

# Print first 24 points (6 hours, adjust as needed)
print("\nForecasted Power Demand for Next 24 Hours (First 24 Points):")
print(forecast_df.head(24))

# Optionally, save to CSV
forecast_df.to_csv("forecasted_power_demand_next_day.csv", index=False)
print("\nData saved to 'forecasted_power_demand_next_day.csv'")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="purple", s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Power Demand (kW)")
plt.ylabel("Predicted Power Demand (kW)")
plt.title("Actual vs Predicted Power Demand (Test Set)")
plt.grid(True)
plt.show()

## 6.8 Scatter Plot
plt.figure(figsize=(10, 6))
errors = y_test - y_pred
sns.histplot(errors, bins=50, kde=True, color="orange")
plt.xlabel("Prediction Error (kW)")
plt.ylabel("Frequency")
plt.title(f"Error Distribution (Test Set, MAE: {mae:.2f} kW)")
plt.grid(True)
plt.show()

## 6.9 Error distribution
plt.figure(figsize=(10, 6))
feature_importances = model.feature_importances_
sns.barplot(x=feature_importances, y=features, hue=features, palette="coolwarm", legend=False)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Power Demand Prediction")
plt.show()

## 6.10 Feature Importance
print("Feature Importances:")
for feat, imp in zip(features, feature_importances):
    print(f"{feat}: {imp:.4f}")

## Sending Data to Thingspeak
#Forecasting for the next 24 hours from current time
last_data = X_test.iloc[-1:].copy()  # Use the last row of X_test as the starting point
predictions = []
# Use current time (May 27, 2025, 11:03 AM +0530)
last_timestamp = pd.to_datetime('2025-05-27 11:03:00+05:30')
time_steps = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=15), periods=96, freq='15min')

# Initialize last_data with current time features
current_hour = 11  # Starting hour (11 AM)
current_day = 27   # Starting day (27th)
current_month = 5  # Starting month (May)
current_weekday = 2  # Tuesday (2 for weekday index, 0=Monday, 6=Sunday)
is_weekend = 0     # Not a weekend (Tuesday)

last_data['Hour'] = current_hour
last_data['Day'] = current_day
last_data['Month'] = current_month
last_data['Weekday'] = current_weekday
last_data['Is_Weekend'] = is_weekend
last_data['Hour_Sin'] = np.sin(2 * np.pi * current_hour / 24)
last_data['Hour_Cos'] = np.cos(2 * np.pi * current_hour / 24)

for i in range(96):
    pred = model.predict(last_data)[0]
    predictions.append(pred)
    last_data['Lag_1h'] = pred
    last_data['Lag_2h'] = last_data['Lag_1h']
    last_data['Lag_24h'] = last_data['Lag_2h']
    last_data['Lag_48h'] = last_data['Lag_24h']
    current_hour += 0.25  # Increment by 15 minutes in hour fraction
    if current_hour >= 24:
        current_hour -= 24
        current_day += 1
        current_weekday = (current_weekday + 1) % 7
        if current_day > 31:  # Assuming May has 31 days
            current_day = 1
            current_month += 1
        is_weekend = 1 if current_weekday in [5, 6] else 0
    last_data['Hour'] = current_hour
    last_data['Day'] = current_day
    last_data['Month'] = current_month
    last_data['Weekday'] = current_weekday
    last_data['Is_Weekend'] = is_weekend
    last_data['Hour_Sin'] = np.sin(2 * np.pi * current_hour / 24)
    last_data['Hour_Cos'] = np.cos(2 * np.pi * current_hour / 24)

# Create forecast_df (remove redundant tz_localize since time_steps is already timezone-aware)
forecast_df = pd.DataFrame({
    'Timestamp': time_steps,  # Already in +05:30 timezone
    'Predicted_Power_Demand_kW': predictions
})

# Print first 24 points
print("\nForecasted Power Demand for Next 24 Hours (First 24 Points):")
print(forecast_df.head(24))

# ThingSpeak upload
write_api_key = '****************'
channel_url = 'https://api.thingspeak.com/update'

for index, row in forecast_df.iterrows():
    payload = {
        'api_key': write_api_key,
        'field7': row['Predicted_Power_Demand_kW'],  # Match your Field 7 Chart
        'created_at': row['Timestamp'].isoformat()
    }
    response = requests.post(channel_url, data=payload)
    if response.status_code == 200:
        print(f"Uploaded: {row['Predicted_Power_Demand_kW']:.2f} kW at {row['Timestamp']}")
    else:
        print(f"Failed to upload: {response.text}")
    time.sleep(15)  # Respect free plan rate limit

print("✅ All predictions uploaded to ThingSpeak!")
model.save_model("D:/PowerDemandPrediction/xgboost_power_model.json")
print("✅ Model trained, evaluated, and saved!")
