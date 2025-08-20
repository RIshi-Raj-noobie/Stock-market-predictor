# Stock Price Direction Prediction Model
# Author: Rishiraj Singh Tomar

# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# 1. Data Acquisition
print("Downloading historical data for AAPL...")
ticker = 'AAPL'
data = yf.download(ticker, start='2018-01-01', end='2023-12-31')
print("Download complete!")
print(f"Dataset shape: {data.shape}")

# 2. Feature Engineering: Create Technical Indicators
print("\nEngineering features...")
df = data.copy()

# Calculate Simple Moving Averages
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# Calculate Price Rate of Change (ROC)
df['ROC'] = df['Close'].pct_change(periods=5)

# Calculate Volatility (Standard Deviation of past 5 days' closing price)
df['Volatility'] = df['Close'].rolling(window=5).std()

# Calculate Relative Strength Index (RSI)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Create the target variable: 1 if next day's close > today's close, else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop rows with NaN values created by indicators
df.dropna(inplace=True)

# 3. Define Features (X) and Target (y)
feature_columns = ['SMA_20', 'SMA_50', 'ROC', 'Volatility', 'RSI', 'Volume']
X = df[feature_columns]
y = df['Target']

# 4. Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# 5. Standardize the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Build and Train the Model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Make Predictions and Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.2%}")

# Detailed performance report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

# 8. Feature Importance
print("\nFeature Importances:")
importances = model.feature_importances_
feature_imp = pd.Series(importances, index=feature_columns).sort_values(ascending=False)
print(feature_imp)

# 9. Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Actual vs Predicted (Sample of 50 points for clarity)
sample_index = np.arange(50)
axes[0, 0].plot(sample_index, y_test.values[sample_index], label='Actual', marker='o')
axes[0, 0].plot(sample_index, y_pred[sample_index], label='Predicted', marker='x')
axes[0, 0].set_title('Actual vs. Predicted Direction (Sample)')
axes[0, 0].set_xlabel('Time Index')
axes[0, 0].set_ylabel('Direction (0=Down, 1=Up)')
axes[0, 0].legend()

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix')

# Plot 3: Feature Importance
sns.barplot(x=feature_imp.values, y=feature_imp.index, ax=axes[1, 0])
axes[1, 0].set_title('Feature Importances')

# Plot 4: Actual Price Chart with SMA
df['Close'].plot(ax=axes[1, 1], label='Close Price', alpha=0.5)
df['SMA_20'].plot(ax=axes[1, 1], label='20-Day SMA')
df['SMA_50'].plot(ax=axes[1, 1], label='50-Day SMA')
axes[1, 1].set_title('Price and Moving Averages')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('assets/results.png') # Save the figure
plt.show()

print("\nAnalysis complete. Visualizations saved.")