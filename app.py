from flask import Flask, jsonify
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Function to fetch daily XAU/USD data
def fetch_xau_data_daily():
    xau_data = yf.download("GC=F", interval="1d", start="2004-01-01")
    xau_data = xau_data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
    xau_data = xau_data.asfreq('D', method='ffill')
    return xau_data

# Function to prepare data and forecast
def forecast_xau_30_days():
    # Fetch and prepare the data
    df = fetch_xau_data_daily()
    df['var_Prev_Open'] = df['Open'].shift(1) - df['Open'].shift(2)
    df['var_Prev_Close'] = df['Close'].shift(1) - df['Close'].shift(2)
    df['var_Prev_High'] = df['High'].shift(1) - df['High'].shift(2)
    df['var_Prev_Low'] = df['Low'].shift(1) - df['Low'].shift(2)
    df['var_Prev_Volume'] = df['Volume'].shift(1) - df['Volume'].shift(2)
    df['var_Close'] = df['Close'] - df['Close'].shift(1)
    df['prev_Close'] = df['Close'].shift(1)

    # Create lagged variables
    for lag in range(1, 31):
        df[f'var_Prev_Close_lag_{lag}'] = df['var_Prev_Close'].shift(lag)
        df[f'var_Prev_Volume_{lag}'] = df['var_Prev_Volume'].shift(lag)
        df[f'var_Prev_High_{lag}'] = df['var_Prev_High'].shift(lag)
        df[f'var_Prev_Low_{lag}'] = df['var_Prev_Low'].shift(lag)
        df[f'var_Prev_Open_{lag}'] = df['var_Prev_Open'].shift(lag)
    df.dropna(inplace=True)

    # Set the end of training period
    train_end = df.index[-31]
    forecast_steps = 30

    # Use ARIMA model for forecasting
    # We will use the 'var_Close' column for the ARIMA model
    model = ARIMA(df.loc[:train_end, 'var_Close'], order=(5, 1, 5))  # ARIMA with order (p,d,q)
    model_fit = model.fit()

    # Make predictions for the next 30 days
    forecast = model_fit.forecast(steps=forecast_steps)

    # Calculate Predicted Close Prices
    pred_close_future_30 = df.loc[train_end:, 'prev_Close'].iloc[-1] + forecast.cumsum().values

    # Get actual closing prices for the last 30 days
    actual_last_30_days = df.loc[train_end:, 'Close'].iloc[:30].values.flatten().tolist()

    # Function to ensure all lists have equal length
    def ensure_equal_length(*args):
        max_length = max(len(arg) for arg in args)
        return [list(arg) + [arg[-1]] * (max_length - len(arg)) for arg in args]

    # Ensure all lists have equal length
    actual_last_30_days, pred_close_future_30 = ensure_equal_length(
        actual_last_30_days,
        pred_close_future_30
    )

    # Return the predicted and actual values as a dictionary
    return {
        "actual_last_30_days": actual_last_30_days,
        "predicted_next_30_days": pred_close_future_30
    }

# Define an endpoint to get predictions
@app.route('/predict-xau', methods=['GET'])
def predict_xau():
    forecast = forecast_xau_30_days()
    return jsonify(forecast)

if __name__ == '__main__':
    app.run(debug=True)
