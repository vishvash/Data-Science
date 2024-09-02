import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the time series data
data = np.array([110,125,133,146,158,172,187,196,210])

# Prepare the data for LSTM
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 3
X, y = prepare_data(data, time_steps)

# Reshape input data to fit LSTM model
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(time_steps, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(X, y, epochs=100, verbose=0)

# Forecast the next 10 digits
forecast = []
last_sequence = X[-1]
for _ in range(10):
    next_digit = model.predict(last_sequence.reshape(1, time_steps, 1))[0][0]
    forecast.append(next_digit)
    last_sequence = np.append(last_sequence[1:], next_digit)

print("Forecasted digits:", forecast)
