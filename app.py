from flask import Flask, jsonify
import requests
from lightgbm import LGBMRegressor
from datetime import datetime, timedelta

app = Flask(__name__)

# Constants
DATA_FETCH_URL = "https://fetch-xau-data.vercel.app/fetch-xau-data"

# Function to fetch and prepare data
def get_xau_data():
    response = requests.get(DATA_FETCH_URL)
    if response.status_code == 200:
        data = response.json()
        
        # Convert date strings to datetime objects and store in a list of dictionaries
        formatted_data = []
        for entry in data:
            formatted_data.append({
                'Date': datetime.strptime(entry['Date'], "%Y-%m-%d"),
                'var_Prev_Close': entry['var_Prev_Close'],
                'Close_GC=F': entry['Close_GC=F']
            })
        
        # Ensure dates are consecutive by filling any missing dates with None values
        formatted_data = fill_missing_dates(formatted_data)
        return formatted_data
    else:
        response.raise_for_status()

# Fill missing dates with None values for 'var_Prev_Close' and 'Close_GC=F'
def fill_missing_dates(data):
    filled_data = []
    start_date = data[0]['Date']
    end_date = data[-1]['Date']
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    data_dict = {entry['Date']: entry for entry in data}
    
    for date in date_range:
        if date in data_dict:
            filled_data.append(data_dict[date])
        else:
            filled_data.append({'Date': date, 'var_Prev_Close': None, 'Close_GC=F': None})
    
    return filled_data

# Function to create lagged features
def create_lagged_features(data, target_column, n_lags):
    for i in range(n_lags, len(data)):
        for lag in range(1, n_lags + 1):
            if data[i - lag][target_column] is not None:
                data[i][f'{target_column}_lag_{lag}'] = data[i - lag][target_column]
            else:
                data[i][f'{target_column}_lag_{lag}'] = 0  # Handle missing values
    # Filter out entries without enough lagged data
    return [entry for entry in data if f'{target_column}_lag_{n_lags}' in entry]

# Forecast function
def forecast_xau_30_days():
    data = get_xau_data()
    target_column = 'var_Prev_Close'
    n_lags = 30
    data = create_lagged_features(data, target_column, n_lags)

    # Prepare training data
    train_data = data[:-n_lags]
    forecast_data = data[-n_lags:]
    
    X_train = [[entry[f'{target_column}_lag_{i}'] for i in range(1, n_lags + 1)] for entry in train_data]
    y_train = [entry[target_column] for entry in train_data]
    X_forecast = [[entry[f'{target_column}_lag_{i}'] for i in range(1, n_lags + 1)] for entry in forecast_data]
    
    # Train LightGBM model
    model = LGBMRegressor(objective='regression', n_estimators=100, random_state=123)
    model.fit(X_train, y_train)
    
    # Make predictions for the next 30 days
    predicted_past_30_days = model.predict(X_forecast)
    
    # Calculate cumulative sum for simulated future closing prices
    last_close = forecast_data[-1]['Close_GC=F']
    pred_close_future_30 = [last_close + sum(predicted_past_30_days[:i+1]) for i in range(len(predicted_past_30_days))]
    
    # Get actual closing prices for the last 30 days
    actual_last_30_days = [entry['Close_GC=F'] for entry in data[-30:] if entry['Close_GC=F'] is not None]

    # Return results in JSON format
    return {
        "actual_last_30_days": actual_last_30_days,
        "predicted_past_30_days": predicted_past_30_days.tolist(),
        "predicted_next_30_days": pred_close_future_30
    }

@app.route('/predict-xau', methods=['GET'])
def predict_xau():
    try:
        forecast = forecast_xau_30_days()
        return jsonify(forecast)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
