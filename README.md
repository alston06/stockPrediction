# Stock Prediction App

This project is a **Stock Prediction App** that leverages historical stock data, machine learning, and Streamlit to provide interactive visualizations and predictions of future stock prices.

## Features

1. **Historical Data Retrieval**
   - Fetches stock data starting from January 1, 2010, using the Yahoo Finance API.
   - Displays a summary of the data with descriptive statistics.

2. **Visualizations**
   - Closing Price vs. Time Chart.
   - Closing Price vs. Time Chart with 100-day and 200-day Moving Averages.

3. **Future Stock Price Prediction**
   - Uses a pre-trained LSTM model (`keras_model.h5`) to predict future stock prices.
   - Displays future predictions for the next 100 days.

4. **Streamlit Interface**
   - Interactive user input for stock ticker symbol.
   - Dynamic visualizations and outputs displayed directly in the app.

## Dependencies

The app requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `datetime`
- `keras`
- `streamlit`
- `scikit-learn`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/stock-prediction-app.git
   cd stock-prediction-app
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the pre-trained LSTM model (`keras_model.h5`) in the project directory.

## Usage

1. Run the app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided local URL (e.g., `http://localhost:8501`).

3. Enter a stock ticker symbol (e.g., `AAPL`) in the input field to visualize historical data and predict future prices.

## Code Explanation

### Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
```

### Retrieve Historical Data
```python
today = datetime.today()
yesterday = today - timedelta(days=1)
start = '2010-01-01'
end = yesterday.strftime('%Y-%m-%d')

st.title('Stock Prediction App')
userInput = st.text_input('Enter a stock ticker:', 'AAPL')
df = yf.download(userInput, start=start, end=end)
```

### Display Descriptive Statistics
```python
st.subheader('Data from 2010 to yesterday')
st.write(df.describe())
```

### Visualization of Historical Data
```python
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)
```

### Visualization with Moving Averages
```python
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df['Close'])
st.pyplot(fig)
```

### Future Predictions
```python
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
```

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
- Yahoo Finance for stock data.
- Streamlit for the interactive web app framework.
- Keras for the deep learning model.

