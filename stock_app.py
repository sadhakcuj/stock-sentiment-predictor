import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os

app = Flask(__name__)

# Load model and scaler
try:
    with open('stock_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('stock_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    print("⚠️ Warning: Model files not found. Run stock_train_model.py first!")

@app.route('/')
def index():
    return render_template('stock_dashboard.html', model_loaded=model_loaded)

@app.route('/api/stock/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    """Get current stock data and prediction"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    ticker = ticker.upper()
    
    try:
        # Get current stock data
        stock = yf.download(ticker, period='1d', progress=False)

        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)
        
        if len(stock) == 0:
            return jsonify({'error': f'Stock {ticker} not found'}), 404
        
        current_price = float(stock['Close'].iloc[-1].item())
        
        # Get historical data for features
        hist = yf.download(ticker, period='60d', progress=False)

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        
        if len(hist) < 30:
            return jsonify({'error': f'Not enough data for {ticker}'}), 400
        
        # Calculate features (same as training)
        daily_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
        ma_7 = hist['Close'].tail(7).mean()
        ma_30 = hist['Close'].tail(30).mean()
        volatility = hist['Close'].pct_change().tail(20).std() * 100
        volume_ma = hist['Volume'].tail(20).mean()
        
        # Sentiment (simulated - in real app, get from news sentiment analysis)
        sentiment_score = 0.65
        news_impact = 0.5
        
        # Prepare features for prediction
        features = np.array([[
            daily_return,
            ma_7,
            ma_30,
            volatility,
            volume_ma,
            sentiment_score,
            news_impact
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = float(model.predict(features_scaled)[0])
        
        # Interpret prediction
        if prediction > 0.5:
            signal = "BUY"
            confidence = min(abs(prediction), 1.0)
        elif prediction < -0.5:
            signal = "SELL"
            confidence = min(abs(prediction), 1.0)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        return jsonify({
            'ticker': ticker,
            'current_price': current_price,
            'prediction_return': round(prediction, 2),
            'signal': signal,
            'confidence': round(confidence * 100, 1),
            'ma_7': round(ma_7, 2),
            'ma_30': round(ma_30, 2),
            'volatility': round(volatility, 2),
            'daily_return': round(daily_return, 2),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'model_loaded': model_loaded,
        'model_type': 'Gradient Boosting Regressor',
        'features': 7,
        'training_period': '2 years',
        'update_frequency': 'Daily'
    })

@app.route('/api/history/<ticker>', methods=['GET'])
def get_price_history(ticker):
    """Get price history for charting"""
    ticker = ticker.upper()
    
    try:
        hist = yf.download(ticker, period='6mo', progress=False)

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]
        closes = [float(c.item()) if hasattr(c, 'item') else float(c)
          for c in hist['Close']]
        
        return jsonify({
            'ticker': ticker,
            'dates': dates,
            'prices': closes
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
