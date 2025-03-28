import os
import nltk
import wikipediaapi
import requests
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from nltk.corpus import wordnet
from fastapi import FastAPI, HTTPException
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Download WordNet dataset (only required once)
nltk.download('wordnet')


# News API Key (replace with your key as needed)
NEWS_API_KEY = "b7af606cdfa0434e9a8293e12911546e"

# Cache for trained LSTM models
MODEL_CACHE = {}

# ------------------------- Stock & Sentiment Endpoints -------------------------

def analyze_stock_news(symbol):
    """Fetch recent news articles for the given symbol and analyze sentiment."""
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    
    if response.get("status") != "ok":
        return {"sentiment": "Neutral", "average_score": 0, "news_articles": []}
    
    articles = response.get("articles", [])[:5]
    sentiments = []
    
    for article in articles:
        content = f"{article.get('title', '')} {article.get('description', '')}"
        sentiment = TextBlob(content).sentiment.polarity
        sentiments.append(sentiment)
    
    avg_sentiment = float(np.mean(sentiments)) if sentiments else 0
    sentiment_result = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
    
    return {
        "sentiment": sentiment_result,
        "average_score": avg_sentiment,
        "news_articles": articles
    }

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

def get_lstm_prediction(symbol):
    """Predict future stock price using an LSTM model with caching."""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="2y")["Close"]

        if len(df) < 60:
            return {"error": "Not enough historical data for prediction"}

        # Use cached model if available
        if symbol in MODEL_CACHE:
            model, scaler = MODEL_CACHE[symbol]
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data = scaler.fit_transform(df.values.reshape(-1, 1))

            X_train, y_train = [], []
            for i in range(60, len(data)):
                X_train.append(data[i-60:i, 0])
                y_train.append(data[i, 0])

            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

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

        return float(predicted_price)
    except Exception as e:
        return {"error": f"Stock prediction failed: {str(e)}"}

@app.get("/stock")
def get_stock_info(symbol: str, goal: str = "investor"):
    """
    Endpoint to retrieve stock details along with news sentiment and a predicted future price.
    The response also includes a basic recommendation.
    """
    details = fetch_stock_details(symbol)
    if isinstance(details, dict) and "error" in details:
        return details
    
    news_sentiment = analyze_stock_news(symbol)
    predicted_price = get_lstm_prediction(symbol)

    if isinstance(predicted_price, dict) and "error" in predicted_price:
        return predicted_price

    recommendation = "Buy" if predicted_price > details["current_price"] else "Hold/Sell"
    static_advice = f"The stock {symbol.upper()} is currently at {details['current_price']}. "
    
    if goal.lower() == "trader":
        static_advice += f"Since you are a trader, consider short-term trends and news. Right now, the sentiment is {news_sentiment['sentiment']}. "
    else:
        static_advice += f"Since you are an investor, focus on long-term fundamentals. The P/E ratio is {details['pe_ratio']}. "
    
    static_advice += f"Based on market trends, moving averages, and sentiment analysis, our model suggests you should {recommendation}."

    return {
        "stock_details": details,
        "news_sentiment": news_sentiment,
        "predicted_price": predicted_price,
        "advice": static_advice
    }

@app.get("/portfolio_advice")
def get_portfolio_advice(symbol: str, quantity: int, goal: str = "investor"):
    """
    Provide personalized portfolio advice.
    For example, 'I have [quantity] shares of [symbol]. Should I sell or hold, and if hold, for how long?'
    """
    details = fetch_stock_details(symbol)
    if isinstance(details, dict) and "error" in details:
        return details

    news_sentiment = analyze_stock_news(symbol)
    predicted_price = get_lstm_prediction(symbol)
    if isinstance(predicted_price, dict) and "error" in predicted_price:
        return predicted_price

    if predicted_price > details["current_price"]:
        decision = "Hold"
        hold_time = "at least 6-12 months" if goal.lower() == "investor" else "until the next short-term market adjustment (a few weeks)"
    else:
        decision = "Sell"
        hold_time = "N/A"

    static_advice = (f"You have {quantity} shares of {symbol.upper()} currently priced at {details['current_price']:.2f}. "
                     f"Our prediction estimates the price could reach {predicted_price:.2f}. Based on this, it is advisable to {decision}. ")
    if decision == "Hold":
        static_advice += f"For your goal as a {goal.lower()}, consider holding for {hold_time}."
    else:
        static_advice += "It might be a good idea to sell your shares now."
    
    return {
        "symbol": symbol.upper(),
        "quantity": quantity,
        "stock_details": details,
        "news_sentiment": news_sentiment,
        "predicted_price": predicted_price,
        "advice": static_advice
    }


# ------------------------- Invest Genius API Endpoints -------------------------

def get_nifty50_symbols():
    """Return a list of NIFTY 50 ticker symbols."""
    nifty_tickers = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "SBIN.NS",
        "BHARTIARTL.NS", "HCLTECH.NS", "ASIANPAINT.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "ITC.NS",
        "LT.NS", "AXISBANK.NS", "WIPRO.NS", "DMART.NS", "MARUTI.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS",
        "TITAN.NS", "TECHM.NS", "HDFCLIFE.NS", "JSWSTEEL.NS", "INDUSINDBK.NS", "M&M.NS", "BAJAJFINSV.NS",
        "NTPC.NS", "POWERGRID.NS", "NESTLEIND.NS", "ONGC.NS", "GRASIM.NS", "ADANIPORTS.NS", "CIPLA.NS",
        "SBILIFE.NS", "HINDALCO.NS", "DRREDDY.NS", "TATAMOTORS.NS", "BRITANNIA.NS", "BPCL.NS",
        "COALINDIA.NS", "EICHERMOT.NS", "DIVISLAB.NS", "IOC.NS", "HEROMOTOCO.NS", "UPL.NS",
        "APOLLOHOSP.NS", "TATASTEEL.NS", "BAJAJ-AUTO.NS", "ADANIENT.NS"
    ]
    return nifty_tickers

@app.get("/")
def home():
    """Home endpoint for the Invest Genius API."""
    return {"message": "Welcome to Invest Genius API"}

@app.get("/top-gainers")
def get_top_gainers():
    """Return the top 10 gaining stocks from the NIFTY 50 list."""
    try:
        current_time = datetime.now()
        stocks_data = []
        nifty_stocks = get_nifty50_symbols()

        for symbol in nifty_stocks:
            stock = yf.Ticker(symbol)
            stock_data = stock.history(period="2d")
            if stock_data.empty:
                continue
            try:
                prev_close = stock_data["Close"].iloc[-2]
                current_price = stock_data["Close"].iloc[-1]
                price_change = ((current_price - prev_close) / prev_close) * 100
            except IndexError:
                continue
            stock_info = {
                "symbol": symbol,
                "company_name": stock.info.get("shortName", "N/A"),
                "price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "logo": stock.info.get("logo_url", "N/A"),
                "timestamp": current_time.isoformat()
            }
            stocks_data.append(stock_info)
        if not stocks_data:
            raise HTTPException(status_code=404, detail="No stock data available.")
        top_gainers = sorted(stocks_data, key=lambda x: x["price_change"], reverse=True)[:10]
        return {"top_gainers": top_gainers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/top-losers")
def get_top_losers():
    """Return the top 10 losing stocks from the NIFTY 50 list."""
    try:
        current_time = datetime.now()
        stocks_data = []
        nifty_stocks = get_nifty50_symbols()
        for symbol in nifty_stocks:
            stock = yf.Ticker(symbol)
            stock_data = stock.history(period="2d")
            if stock_data.empty:
                continue
            try:
                prev_close = stock_data["Close"].iloc[-2]
                current_price = stock_data["Close"].iloc[-1]
                price_change = ((current_price - prev_close) / prev_close) * 100
            except IndexError:
                continue
            stock_info = {
                "symbol": symbol,
                "company_name": stock.info.get("shortName", "N/A"),
                "price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "logo": stock.info.get("logo_url", "N/A"),
                "timestamp": current_time.isoformat()
            }
            stocks_data.append(stock_info)
        if not stocks_data:
            raise HTTPException(status_code=404, detail="No stock data available.")
        top_losers = sorted(stocks_data, key=lambda x: x["price_change"])[:10]
        return {"top_losers": top_losers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/all-stocks")
def get_all_stocks():
    """Return all NIFTY 50 stocks with their latest details."""
    try:
        current_time = datetime.now()
        stocks_data = []
        nifty_stocks = get_nifty50_symbols()
        for symbol in nifty_stocks:
            stock = yf.Ticker(symbol)
            stock_data = stock.history(period="2d")
            if stock_data.empty:
                continue
            try:
                prev_close = stock_data["Close"].iloc[-2]
                current_price = stock_data["Close"].iloc[-1]
                price_change = ((current_price - prev_close) / prev_close) * 100
            except IndexError:
                continue
            stock_info = {
                "symbol": symbol,
                "company_name": stock.info.get("shortName", "N/A"),
                "price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "logo": stock.info.get("logo_url", "N/A"),
                "timestamp": current_time.isoformat()
            }
            stocks_data.append(stock_info)
        if not stocks_data:
            raise HTTPException(status_code=404, detail="No stock data available.")
        return {"all_stocks": stocks_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/indian-indices")
def get_indian_indices():
    """Return current values for key Indian indices."""
    try:
        current_time = datetime.now()
        index_data = {}
        indices_info = {
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN",
            "NIFTY BANK": "^NSEBANK",
            "NIFTY IT": "^CNXIT",
            "NIFTY MIDCAP 50": "^NSEMDCP50",
            "NIFTY 100": "^CNX100",
            "NIFTY FINANCIAL SERVICES (FINNIFTY)": "^CNXFIN",
            "INDIA VIX": "^INDIAVIX"
        }
        for name, symbol in indices_info.items():
            try:
                data = yf.Ticker(symbol).history(period="1d")
                index_data[name] = {
                    "Current Value": round(data["Close"].iloc[-1], 2)
                }
            except Exception as e:
                index_data[name] = f"Error: {e}"
        if not index_data:
            raise HTTPException(status_code=404, detail="No index data available.")
        return {"indian_indices": index_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
