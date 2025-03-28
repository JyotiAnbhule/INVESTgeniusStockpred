import os
from fastapi import FastAPI
import yfinance as yf
import requests
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import uvicorn

# Disable GPU usage if you're not using a GPU to avoid TensorFlow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage

app = FastAPI()

MODEL_CACHE = {}
NEWS_API_KEY = "your_news_api_key_here"  # Replace this with a valid API key

def fetch_stock_details(symbol):
    """Fetch stock details and calculate technical indicators."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="6mo")
        
        if hist.empty or "Close" not in hist:
            return {"error": "Invalid stock symbol or no data available"}
        
        current_price = stock.history(period="1d")["Close"].iloc[-1] if not stock.history(period="1d").empty else None
        if current_price is None:
            return {"error": "Current stock price data not available"}
        
        info = stock.info
        
        return {
            "current_price": float(current_price),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "sma_50": float(hist["Close"].rolling(window=50).mean().iloc[-1]) if len(hist) >= 50 else "N/A",
            "sma_200": float(hist["Close"].rolling(window=200).mean().iloc[-1]) if len(hist) >= 200 else "N/A",
        }
    except Exception as e:
        return {"error": f"Failed to fetch stock details: {str(e)}"}

def analyze_stock_news(symbol):
    """Fetch recent news articles and analyze sentiment."""
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    
    if response.get("status") != "ok":
        return 0
    
    articles = response.get("articles", [])[:5]
    sentiments = [TextBlob(f"{a.get('title', '')} {a.get('description', '')}").sentiment.polarity for a in articles]
    return float(np.mean(sentiments)) if sentiments else 0

def get_lstm_prediction(symbol):
    """Predict future stock price using an LSTM model considering news sentiment."""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="2y")["Close"]
        if len(df) < 60:
            return {"error": "Not enough historical data for prediction"}
        
        news_sentiment = analyze_stock_news(symbol)  # Include sentiment in prediction
        
        if symbol in MODEL_CACHE:
            model, scaler = MODEL_CACHE[symbol]
        else:
            # Training the LSTM model if not in cache
            scaler = MinMaxScaler(feature_range=(0, 1))
            data = scaler.fit_transform(df.values.reshape(-1, 1))
            
            X_train, y_train = [], []
            for i in range(60, len(data)):
                X_train.append(data[i-60:i, 0])
                y_train.append(data[i, 0])
            
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                tf.keras.layers.LSTM(50, return_sequences=False),
                tf.keras.layers.Dense(25),
                tf.keras.layers.Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
            MODEL_CACHE[symbol] = (model, scaler)
        
        last_60_days = df[-60:].values.reshape(-1, 1)
        last_60_days_scaled = scaler.transform(last_60_days)
        future_input = last_60_days_scaled.reshape(1, 60, 1)
        predicted_price = scaler.inverse_transform(model.predict(future_input))[0][0]
        
        # Adjust prediction based on sentiment
        adjusted_prediction = predicted_price + (news_sentiment * predicted_price * 0.01)
        
        return float(adjusted_prediction)
    except Exception as e:
        return {"error": f"Stock prediction failed: {str(e)}"}

@app.get("/stock")
def get_stock_info(symbol: str):
    """Retrieve stock details along with a predicted future price."""
    details = fetch_stock_details(symbol)
    if "error" in details:
        return details
    
    predicted_price = get_lstm_prediction(symbol)
    if isinstance(predicted_price, dict) and "error" in predicted_price:
        return predicted_price
    
    recommendation = "Buy" if predicted_price > details["current_price"] else "Hold/Sell"
    return {
        "stock_details": details,
        "predicted_price": predicted_price,
        "advice": f"Based on analysis, it is recommended to {recommendation}."
    }

# Start the FastAPI app with dynamic port using the environment variable
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use the PORT environment variable, default to 8000 if not set
    uvicorn.run(app, host="0.0.0.0", port=port)
