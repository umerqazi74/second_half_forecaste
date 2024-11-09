from flask import Flask, jsonify
import requests
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from xgboost import XGBRegressor

app = Flask(__name__)

# URL of the first part endpoint
DATA_FETCH_URL = "https://fetch-xau-data.vercel.app/fetch-xau-data"

# Function to get data from the first part
def get_xau_data():
    response = requests.get(DATA_FETCH_URL)
    if response.status_code == 200:
        data = pd.DataFrame(response.json())
        # Re-convert 'Date' column to datetime after fetching
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # Explicitly set frequency if not already set
        if data.index.freq is None:
            data = data.asfreq('D')
        return data
    else:
        response.raise_for_status()


# Function to prepare data and forecast
def forecast_xau_30_days():
    df = get_xau_data()

    # Additional feature engineering
    df['var_Prev_Open'] = df['Open_GC=F'].shift(1) - df['Open_GC=F'].shift(2)
    df['var_Prev_High'] = df['High_GC=F'].shift(1) - df['High_GC=F'].shift(2)
    df['var_Prev_Low'] = df['Low_GC=F'].shift(1) - df['Low_GC=F'].shift(2)
    df['var_Prev_Volume'] = df['Volume_GC=F'].shift(1) - df['Volume_GC=F'].shift(2)
    df['var_Close'] = df['Close_GC=F'] - df['Close_GC=F'].shift(1)
    df['prev_Close'] = df['Close_GC=F'].shift(1)

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

    # Define and fit the forecaster
    forecaster_xgb_daily = ForecasterAutoreg(
        regressor=XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=123),
        lags=30
    )
    forecaster_xgb_daily.fit(
        y=df.loc[:train_end, 'var_Close'],
        exog=df.loc[:train_end, [col for col in df.columns if 'var_Prev' in col]]
    )

    # Align `exog` data to start one day after training ends
    exog_for_past_30 = df.loc[train_end + pd.Timedelta(days=1):, [col for col in df.columns if 'var_Prev' in col]].iloc[:30]
    exog_for_next_30 = df.loc[train_end + pd.Timedelta(days=1):, [col for col in df.columns if 'var_Prev' in col]].iloc[-30:]

    # Make predictions
    predicted_past_30_days = forecaster_xgb_daily.predict(steps=forecast_steps, exog=exog_for_past_30)
    predicted_next_30_days = forecaster_xgb_daily.predict(steps=forecast_steps, exog=exog_for_next_30)

    # Calculate Predicted Close Prices
    pred_close_past_30 = df.loc[train_end:, 'prev_Close'].iloc[:30].values + predicted_past_30_days.values
    pred_close_future_30 = df.loc[train_end:, 'prev_Close'].iloc[-1] + predicted_next_30_days.cumsum().values

    # Get actual closing prices for the last 30 days
    actual_last_30_days = df.loc[train_end:, 'Close_GC=F'].iloc[:30].values.flatten().tolist()

    # Function to ensure all lists have equal length
    def ensure_equal_length(*args):
        max_length = max(len(arg) for arg in args)
        return [list(arg) + [arg[-1]] * (max_length - len(arg)) for arg in args]

    # Ensure all lists have equal length
    actual_last_30_days, pred_close_past_30, pred_close_future_30 = ensure_equal_length(
        actual_last_30_days,
        pred_close_past_30,
        pred_close_future_30
    )

    # Return the predicted and actual values as a dictionary
    return {
    "actual_last_30_days": actual_last_30_days,
    "predicted_past_30_days": pred_close_past_30,  # Remove .tolist()
    "predicted_next_30_days": pred_close_future_30  # Remove .tolist()
}

# Define an endpoint to get predictions
@app.route('/predict-xau', methods=['GET'])
def predict_xau():
    forecast = forecast_xau_30_days()
    return jsonify(forecast)

if __name__ == '__main__':
    app.run(debug=True)
