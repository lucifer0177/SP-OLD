from flask import Flask, render_template, request, jsonify, logging
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def get_market_data():
    """Fetch top gainers, losers, most active and trending tickers"""
    try:
        # Get actual market movers data from Yahoo Finance
        app.logger.info("Fetching market data from Yahoo Finance")
        
        # Define tickers to track
        tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC']
        
        # Initialize lists to store results
        top_gainers = []
        top_losers = []
        most_active = []
        trending_tickers = []
        
        # Process each ticker
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                change = info.get('regularMarketChangePercent', 0)
                volume = info.get('regularMarketVolume', 0)
                
                # Categorize based on performance
                if change > 0:
                    top_gainers.append({'symbol': ticker, 'change': change})
                elif change < 0:
                    top_losers.append({'symbol': ticker, 'change': change})
                
                # Track volume and trending status
                most_active.append({'symbol': ticker, 'volume': volume})
                trending_tickers.append({'symbol': ticker, 'change': change})
                
            except Exception as e:
                app.logger.error(f"Error processing {ticker}: {str(e)}")
                continue

        # Sort and limit results
        result = {
            'topGainers': sorted(top_gainers, key=lambda x: x['change'], reverse=True)[:5],
            'topLosers': sorted(top_losers, key=lambda x: x['change'])[:5],
            'mostActive': sorted(most_active, key=lambda x: x['volume'], reverse=True)[:5],
            'trendingTickers': sorted(trending_tickers, key=lambda x: abs(x['change']), reverse=True)[:5]
        }
        
        # Validate results
        if not all(isinstance(x, list) for x in result.values()):
            raise ValueError("Invalid data format returned")
            
        app.logger.info("Successfully fetched market data")
        return result

    except Exception as e:
        app.logger.error(f"Error fetching market data: {str(e)}")
        return None

@app.route('/market-data')
def market_data():
    """Endpoint to serve market data"""
    data = get_market_data()
    if data:
        return jsonify(data)
    return jsonify({'error': 'Failed to fetch market data'}), 500


# Configure logging
import logging
logging.basicConfig(level=logging.INFO)

def fetch_stock_data(symbol, period='2y'):
    """Fetch historical stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if df.empty:
            raise Exception(f"No data found for symbol {symbol}. Please verify the stock symbol.")
        return df
    except Exception as e:
        app.logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
        raise Exception(f"Error fetching stock data: {str(e)}")

def prepare_features(df):
    """Prepare features for the ML model"""
    try:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'] = calculate_macd(df['Close'])
        df['Target'] = df['Close'].shift(-1)  # Next day's closing price
        df = df.dropna()
        return df
    except Exception as e:
        app.logger.error(f"Error preparing features: {str(e)}")
        raise

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        app.logger.error(f"Error calculating RSI: {str(e)}")
        raise

def calculate_macd(prices, slow=26, fast=12):
    """Calculate MACD technical indicator"""
    try:
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2
    except Exception as e:
        app.logger.error(f"Error calculating MACD: {str(e)}")
        raise

def train_model(df):
    """Train the Random Forest model"""
    try:
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Open', 'High', 'Low', 'Close', 'Volume']
        X = df[features]
        y = df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler, features
    except Exception as e:
        app.logger.error(f"Error training model: {str(e)}")
        raise

def generate_explanation(model, X, features):
    """Generate detailed SHAP explanations for the model's predictions"""
    try:
        app.logger.info("Initializing SHAP explainer")
        explainer = shap.TreeExplainer(model)
        app.logger.info("SHAP explainer initialized successfully")
        
        app.logger.info("Calculating SHAP values")
        shap_values = explainer.shap_values(X)
        if shap_values is None:
            raise ValueError("SHAP values calculation returned None")
        app.logger.info(f"SHAP values calculated for {len(features)} features")
        
        # Log first few SHAP values for verification
        app.logger.debug(f"Sample SHAP values: {shap_values[:2]}")
        
        # Calculate feature importance based on SHAP values
        feature_importance = {}

        feature_descriptions = {
            'SMA_20': '20-day Simple Moving Average - shows the average price over the last 20 days',
            'SMA_50': '50-day Simple Moving Average - shows the average price over the last 50 days',
            'RSI': 'Relative Strength Index - measures the speed and change of price movements (30-70 range)',
            'MACD': 'Moving Average Convergence Divergence - shows the relationship between two moving averages',
            'Open': 'Opening price of the stock for the day',
            'High': 'Highest price of the stock during the day',
            'Low': 'Lowest price of the stock during the day',
            'Close': 'Closing price of the stock for the day',
            'Volume': 'Number of shares traded during the day'
        }

        # Calculate importance and create detailed explanations
        detailed_explanations = []
        for idx, feature in enumerate(features):
            importance = float(abs(shap_values[:, idx].mean()))
            feature_importance[feature] = importance
            detailed_explanations.append({
                'feature': feature,
                'importance': importance,
                'description': feature_descriptions.get(feature, ''),
                'impact': shap_values[:, idx].mean()
            })
        
        # Sort features by importance
        detailed_explanations.sort(key=lambda x: x['importance'], reverse=True)
        
        # Create explanation data structure
        explanation_data = {
            'feature_importance': feature_importance,
            'shap_values': shap_values.tolist(),
            'detailed_explanations': detailed_explanations,
            'overall_impact': {
                'positive': float(sum(x['impact'] for x in detailed_explanations if x['impact'] > 0)),
                'negative': float(sum(x['impact'] for x in detailed_explanations if x['impact'] < 0))
            }
        }
        
        # Validate explanation data
        if not explanation_data.get('detailed_explanations'):
            raise ValueError("Detailed explanations not generated")
        if not explanation_data.get('overall_impact'):
            raise ValueError("Overall impact not calculated")
            
        app.logger.info(f"Generated explanation data with {len(detailed_explanations)} features")
        app.logger.info(f"Overall impact - Positive: {explanation_data['overall_impact']['positive']}, Negative: {explanation_data['overall_impact']['negative']}")
        app.logger.debug(f"Full explanation data: {explanation_data}")
        
        return explanation_data

    except Exception as e:
        app.logger.error(f"Error generating explanation: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stock-of-the-day', methods=['GET'])
def stock_of_the_day():
    """Endpoint to suggest a stock of the day based on predictions."""
    # Logic to determine stock of the day based on predictions
    market_data = get_market_data()
    top_gainers = market_data['topGainers']
    
    app.logger.info(f"Top gainers: {top_gainers}")
    if top_gainers:
        stock_of_the_day = top_gainers[0]['symbol']  # Get the top gainer
        app.logger.info(f"Selected stock of the day: {stock_of_the_day}")
        predicted_profit = top_gainers[0]['change']  # Use the change as predicted profit

    else:
        stock_of_the_day = "N/A"
        predicted_profit = 0.0

    return jsonify({
        'stock_of_the_day': stock_of_the_day,
        'predicted_profit': predicted_profit
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        symbol = None
        historical_data = None
        app.logger.info("Received prediction request")

        data = request.get_json()
        if not data:
            app.logger.error("No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
            
        symbol = data.get('symbol', '').upper()
        app.logger.info(f"Processing prediction for symbol: {symbol}")
        
        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        # Fetch and prepare data
        historical_data = fetch_stock_data(symbol, period='1mo')  # Fetch historical data for the last month


        app.logger.info("Fetching data for symbol: {}".format(symbol))

        df = fetch_stock_data(symbol)
        app.logger.info("Preparing features for symbol: {}".format(symbol))

        df = prepare_features(df)
        
        # Train model
        app.logger.info("Training model for symbol: {}".format(symbol))

        model, scaler, features = train_model(df)
        
        # Prepare latest data for prediction
        latest_data = df[features].iloc[-1:]
        latest_scaled = scaler.transform(latest_data)
        
        # Make prediction
        app.logger.info("Making prediction for symbol: {}".format(symbol))

        prediction = model.predict(latest_scaled)[0]
        
        # Generate explanation
        app.logger.info(f"Generating explanation for {symbol}")
        explanation = generate_explanation(model, latest_scaled, features)
        
        # Get real-time current price
        try:
            current_market_data = yf.Ticker(symbol).history(period='1d')
            if not current_market_data.empty:
                current_price = float(current_market_data['Close'].iloc[-1])
            else:
                current_price = float(df['Close'].iloc[-1])
                app.logger.warning(f"Using historical close price for {symbol} as real-time data unavailable")
        except Exception as e:
            current_price = float(df['Close'].iloc[-1])
            app.logger.error(f"Error fetching real-time price: {str(e)}. Using historical close price")

        # Calculate prediction metrics
        price_change = float(prediction - current_price)
        price_change_percent = float((price_change / current_price) * 100)
        
        buy_price = round(prediction * 0.98, 2)  # 2% lower than predicted price
        sell_price = round(prediction * 1.02, 2)  # 2% higher than predicted price
        
        response = {
            'historical_data': historical_data['Close'].tolist(),  # Include historical closing prices
            'dates': historical_data.index.strftime('%Y-%m-%d').tolist(),  # Include corresponding dates

            'symbol': symbol,
            'current_price': round(current_price, 2),
            'predicted_price': round(prediction, 2),
            'buy_price': buy_price,
            'sell_price': sell_price,
            'price_change': round(price_change, 2),
            'price_change_percent': round(price_change_percent, 2),
            'explanation': explanation,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'buy_sell_explanation': f"Suggested Buy Price: {buy_price} (2% below predicted price), Suggested Sell Price: {sell_price} (2% above predicted price)"
        }

        
        return jsonify(response)



    
    except Exception as e:
        error_msg = f"Error in prediction for {symbol}: {str(e)}"
        app.logger.error(error_msg)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
