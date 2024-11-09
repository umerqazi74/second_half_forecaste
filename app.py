from flask import Flask, jsonify
import pandas as pd
import requests
from lightgbm import LGBMRegressor  # Import LGBMRegressor from LightGBM
import numpy as np

app = Flask(__name__)

# Constants
DATA_FETCH_URL = "https://fetch-xau-data.vercel.app/fetch-xau-data"

# Function to fetch data from the first part
def get_xau_data():
    response = requests.get(DATA_FETCH_URL)
    if response.status_code == 200:
        data = pd.DataFrame(response.json())
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        if data.index.freq is None:
            data = data.asfreq('D')
        return data
    else:
        response.raise_for_status()

# Function to create lagged features
def create_lagged_features(df, target_column, n_lags):
    for lag in range(1, n_lags + 1):
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    df.dropna(inplace=True)
    return df

def forecast_xau_30_days():
    # Fetch the data
    df = get_xau_data()
    
    # Define target and lagged features
    target_column = 'var_Prev_Close'
    n_lags = 30
    df = create_lagged_features(df, target_column, n_lags)
    
    # Separate the data into training and forecasting sets
    train_data = df.iloc[:-n_lags]
    forecast_data = df.iloc[-n_lags:]
    
    # Split features and target variable
    X_train = train_data[[f'{target_column}_lag_{i}' for i in range(1, n_lags + 1)]]
    y_train = train_data[target_column]
    X_forecast = forecast_data[[f'{target_column}_lag_{i}' for i in range(1, n_lags + 1)]]

    # Define and train the LightGBM model
    model = LGBMRegressor(objective='regression', n_estimators=100, random_state=123)
    model.fit(X_train, y_train)
    
    # # Make predictions for the next 30 days
    # predicted_past_30_days = model.predict(X_forecast)
    # pred_close_future_30 = forecast_data['Close_GC=F'].iloc[-1] + np.cumsum(predicted_past_30_days)

    # Convert the predictions to Python float to make them JSON serializable
    # predicted_past_30_days = predicted_past_30_days.astype(float).tolist()
    pred_close_future_30 = pred_close_future_30.astype(float).tolist()

    # Get actual closing prices for the last 30 days
    actual_last_30_days = df['Close_GC=F'].iloc[-30:].astype(float).tolist()

    # Return the results in JSON format
    return {
        "actual_last_30_days": actual_last_30_days,
        # "predicted_past_30_days": predicted_past_30_days,
        "predicted_next_30_days": pred_close_future_30
    }

@app.route('/predict-xau', methods=['GET'])
def predict_xau():
    forecast = forecast_xau_30_days()
    return jsonify(forecast)


if __name__ == '__main__':
    app.run(debug=True)
