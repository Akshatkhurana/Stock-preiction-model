import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense
import joblib
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def train_model(stock_name):
    # Fetch and preprocess stock data
    data = yf.download(stock_name, start="2010-01-01", end="2023-01-01")
    data['Close'] = data['Close'].ffill()
    close_prices = data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Prepare data for LSTM
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build the LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save the trained model
    model.save(f"{stock_name}_model.h5")
    return scaler

if __name__ == "__main__":
    scaler = train_model('MSFT')
    joblib.dump(scaler, 'scaler.save')  # Save the scaler for predictions
