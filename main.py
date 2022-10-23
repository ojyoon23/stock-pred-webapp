import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import plotly.express as px
import pystan
from datetime import datetime
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import streamlit as st

start = '2016-01-01'
end = datetime.now()

st.title('Stock Trend Forecasting App')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)
df.head()
df = df.reset_index()

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

st.subheader('Raw data for the last five days')
st.write(df.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df['Date'],y=df['Open'], name = 'stock open'))
    fig.add_trace(go.Scatter(x = df['Date'], y=df['Close'], name = 'stock close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Visualizations with MA
ma_day = [20, 100]

for day in ma_day:
    col_name = f"MA for {day} days"
    df[col_name] = df['Close'].rolling(day).mean()

st.subheader('Closing Price with Moving Average')
fig = px.line(df[['Close', 'MA for 20 days', 'MA for 100 days']], width=800, height=450)
st.plotly_chart(fig)


#Forecasting
df_train = df[['Date', 'Close']]
df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write('Forecast data table')
st.write(forecast.tail())

fig1 = plot_plotly(m, forecast)
fig1.layout.update(title_text="Forcast Data")
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

st.write('These graphs shows the changes of stock price trend based on day of week and year')

if __name__ == '__main__':
  main()