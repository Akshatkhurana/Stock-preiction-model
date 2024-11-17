import os
import sys
import numpy as np
import yfinance as yf
from tf_keras.models import load_model
import joblib
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
def predict_stock(stock_name, days):
    # Get the base directory where the script is running
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct paths to model and scaler in the "model" folder
    model_path = os.path.join(base_dir, "../model", f"{stock_name}_model.h5")
    scaler_path = os.path.join(base_dir, "../model", "scaler.save")

    # Load model and scaler
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Fetch the last 60 days of stock prices
    try:
        stock_data = yf.download(stock_name, period="6mo", interval="1d")  # Fetch last 70 days to account for market holidays
        last_60_days = stock_data['Close'].dropna().values[-60:]  # Get the last 60 closing prices
        if len(last_60_days) < 60:
            raise ValueError("Not enough data to generate predictions (less than 60 days of data available).")
        last_60_days = last_60_days.reshape(-1, 1)  # Reshape for scaler
    except Exception as e:
        raise RuntimeError(f"Failed to fetch stock data for {stock_name}: {str(e)}")

    # Scale the data
    scaled_data = scaler.transform(last_60_days)

    # Prepare data for prediction
    X_test = np.array([scaled_data])
    predictions = []
    for _ in range(int(days)):
        pred = model.predict(X_test)
        predictions.append(pred[0][0])
        # Update input data with the latest prediction
        X_test = np.roll(X_test, -1, axis=1)
        X_test[0, -1] = pred[0][0]

    # Inverse scale predictions to original stock price range
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    predictions = [round(float(pred[0]), 8) for pred in predictions]
    return predictions

if __name__ == "__main__":
    try:
        stock_name = sys.argv[1]
        days = int(sys.argv[2])
        predictions = predict_stock(stock_name, days)
        print(json.dumps(predictions))  # Only print JSON
    except Exception as e:
        print(json.dumps({"error": str(e)}))  # Return error in JSON format
        sys.exit(1)

