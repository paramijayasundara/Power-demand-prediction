import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1️⃣ Load the dataset
file_path = r"D:/PowerDemandPrediction/electricity_data.csv"
df = pd.read_csv(file_path)

# 2️⃣ Preprocess the data
# Drop unnecessary columns (You don't measure Temperature, Humidity, etc.)
columns_to_remove = ["Solar Power (kW)", "Wind Power (kW)", "Grid Supply (kW)", 
                     "Voltage Fluctuation (%)", "Overload Condition", "Transformer Fault",
                     "Electricity Price (USD/kWh)","Predicted Load (kW)"]
df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

# Convert Timestamp to DateTime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.set_index("Timestamp")

# Check for missing values
df = df.dropna()

# 3️⃣ Feature Selection (Predict power demand using relevant features)
X = df.drop(columns=["Predicted Load (kW)"])  # Independent variables (features)
y = df["Predicted Load (kW)"]  # Dependent variable (target)

# 4️⃣ Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Train the XGBoost Model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 6️⃣ Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# 📊 7️⃣ Visualization with Seaborn
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")

# Actual vs Predicted
plt.subplot(1, 2, 1)
sns.lineplot(x=range(len(y_test)), y=y_test.values, label="Actual Power Demand", color="blue")
sns.lineplot(x=range(len(y_pred)), y=y_pred, label="Predicted Power Demand", color="red", linestyle="dashed")
plt.xlabel("Samples")
plt.ylabel("Power Demand (kW)")
plt.title("Actual vs Predicted Power Demand")
plt.legend()

# Error Distribution
plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred, bins=30, kde=True, color="purple")
plt.xlabel("Prediction Error (kW)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")

plt.tight_layout()
plt.show()

# 📈 Feature Importance
plt.figure(figsize=(8, 5))
feature_importances = model.feature_importances_
sns.barplot(x=feature_importances, y=X.columns, palette="coolwarm")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Power Demand Prediction")
plt.show()

# ✅ Save the model for future use
model.save_model("D:/PowerDemandPrediction/xgboost_power_model.json")

print("✅ Model training completed and saved!")