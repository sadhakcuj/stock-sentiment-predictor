import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STOCK SENTIMENT PREDICTOR - MODEL TRAINING")
print("=" * 70)

# Step 1: Download historical stock data
print("\n📥 Downloading historical stock data...")

tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN']
all_data = []

for ticker in tickers:
    try:
        # Download 2 years of data
        stock = yf.download(ticker, period='2y', progress=False, interval='1d')
        # FIX MULTI-INDEX COLUMNS
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)
        
        # Calculate features
        stock['Daily_Return'] = stock['Close'].pct_change() * 100
        stock['MA_7'] = stock['Close'].rolling(window=7).mean()
        stock['MA_30'] = stock['Close'].rolling(window=30).mean()
        stock['Volatility'] = stock['Daily_Return'].rolling(window=20).std()
        stock['Volume_MA'] = stock['Volume'].rolling(window=20).mean()
        stock['Ticker'] = ticker
        
        # Drop NaN values
        #stock = stock.dropna()
        
        all_data.append(stock)
        print(f"✅ Downloaded {len(stock)} days of data for {ticker}")
    
    except Exception as e:
        print(f"⚠️  Error downloading {ticker}: {e}")

# Combine all data
combined_data = pd.concat(all_data, axis=0)
print(f"\n✅ Total dataset: {len(combined_data)} rows")

# Step 2: Create target variable (predict next day return)
print("\n🎯 Creating target variable...")

combined_data['Next_Day_Return'] = combined_data.groupby('Ticker')['Daily_Return'].shift(-1)
print(combined_data.isna().sum())
combined_data = combined_data.dropna()

print(f"Samples with targets: {len(combined_data)}")

# Step 3: Prepare features
print("\n🔍 Preparing features...")

feature_columns = [
    'Daily_Return', 'MA_7', 'MA_30', 'Volatility', 'Volume_MA'
]

# Add synthetic sentiment features (simulating news sentiment)
np.random.seed(42)
combined_data['Sentiment_Score'] = np.random.uniform(0.3, 0.9, len(combined_data))
combined_data['News_Impact'] = np.random.uniform(-2, 2, len(combined_data))

feature_columns.extend(['Sentiment_Score', 'News_Impact'])

X = combined_data[feature_columns].copy()
y = combined_data['Next_Day_Return'].copy()

print(f"Features: {feature_columns}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Step 4: Scale features
if len(X) == 0:
    print("❌ No samples available after preprocessing")
    exit()
print("\n📊 Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features scaled (mean=0, std=1)")

# Step 5: Split data
print("\n🔀 Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Step 6: Train model
print("\n🤖 Training ensemble model...")
print("   Using Gradient Boosting Regressor...")

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)
print("✅ Model training complete!")

# Step 7: Evaluate model
print("\n📈 Model Evaluation")
print("=" * 70)

# Training performance
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

print("\nTraining Set:")
print(f"  R² Score:  {train_r2:.4f}")
print(f"  RMSE:      {train_rmse:.4f}")
print(f"  MAE:       {train_mae:.4f}")

# Test performance
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\nTest Set:")
print(f"  R² Score:  {test_r2:.4f}")
print(f"  RMSE:      {test_rmse:.4f}")
print(f"  MAE:       {test_mae:.4f}")

# Accuracy for direction prediction
y_test_direction = (y_test > 0).astype(int)
y_test_pred_direction = (y_test_pred > 0).astype(int)
direction_accuracy = (y_test_direction == y_test_pred_direction).mean()

print(f"\nDirection Accuracy: {direction_accuracy:.1%}")
print("  (Correctly predicts up/down movement)")

# Feature importance
print("\n🎯 Feature Importance:")
importances = model.feature_importances_
for feat, imp in sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True):
    print(f"  {feat:20s}: {imp:.4f}")

# Step 8: Save model
print("\n💾 Saving model and scaler...")

with open('stock_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('stock_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model saved to stock_model.pkl")
print("✅ Scaler saved to stock_scaler.pkl")

# Step 9: Summary statistics
print("\n" + "=" * 70)
print("📊 MODEL SUMMARY")
print("=" * 70)
print(f"Algorithm: Gradient Boosting Regressor")
print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")
print(f"Features: {len(feature_columns)}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Direction Prediction Accuracy: {direction_accuracy:.1%}")
print("=" * 70)
print("✨ TRAINING COMPLETE - Ready for predictions!")
print("=" * 70)
