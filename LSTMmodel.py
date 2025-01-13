import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

today = datetime.today()
yesterday = today - timedelta(days=1)
start = '2010-01-01'
end = yesterday.strftime('%Y-%m-%d')

st.title('Stock Prediction App')

userInput = st.text_input('Enter a stock ticker:', 'AAPL')
df = yf.download(userInput, start=start, end=end)

st.subheader('Data from 2010 to yesterday')
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with  & 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df['Close'])
st.pyplot(fig)

model = load_model('keras_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df[['Close']])

future_days = 100
last_sequence = data_scaled[-100:]
future_predictions = []
current_input = last_sequence


for _ in range(future_days):
    prediction = model.predict(current_input[np.newaxis, :, :])
    future_predictions.append(prediction[0, 0])
    current_input = np.append(current_input[1:], prediction, axis=0)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
historical_dates = df.index
future_dates = pd.date_range(historical_dates[-1] + timedelta(days=1), periods=future_days)

st.subheader('Future Price Prediction')
fig = plt.figure(figsize=(12, 6))
plt.plot(historical_dates, df['Close'], label='Historical Prices')
plt.plot(future_dates, future_predictions, label='Predicted Future Prices', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)
