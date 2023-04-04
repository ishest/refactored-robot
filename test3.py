import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm

import yfinance as yf

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pyfolio as pf
import numpy as np
import io
import contextlib
def generate_returns_series(df, predictions):
    df_backtest = df.iloc[-len(predictions):].copy()
    df_backtest['Predictions'] = predictions
    df_backtest['Buy_Signal'] = (df_backtest['Predictions'] == 1)

    returns = []

    for i in range(len(df_backtest) - 1):
        if df_backtest.iloc[i]['Buy_Signal']:
            daily_return = (df_backtest.iloc[i + 1]['Close'] - df_backtest.iloc[i]['Close']) / df_backtest.iloc[i]['Close']
            returns.append(daily_return)
        else:
            returns.append(0)

    return pd.Series(returns, index=df_backtest.index[:-1])

def display_backtest_statistics(returns):
    st.subheader("Backtest Statistics")

    # Calculate and display summary statistics
    st.write("Summary statistics:")
    summary_stats = pf.timeseries.perf_stats(returns)
    st.write(summary_stats)

    # Plot cumulative returns
    st.subheader("Cumulative Returns")
    fig_cumulative_returns, _ = plt.subplots()
    pf.plot_returns(returns, ax=plt.gca())
    st.pyplot(fig_cumulative_returns)

    # Plot rolling volatility
    st.subheader("Rolling Volatility")
    fig_rolling_volatility, _ = plt.subplots()
    pf.plotting.plot_rolling_volatility(returns, ax=plt.gca())
    st.pyplot(fig_rolling_volatility)

    # Plot monthly returns heatmap
    st.subheader("Monthly Returns Heatmap")
    fig_monthly_heatmap, _ = plt.subplots()
    pf.plotting.plot_monthly_returns_heatmap(returns, ax=plt.gca())
    st.pyplot(fig_monthly_heatmap)

# def display_backtest_statistics(returns):
#     st.subheader("Backtest Statistics")
#
#     # Calculate and display summary statistics
#     st.write("Summary statistics:")
#     summary_stats = pf.timeseries.perf_stats(returns)
#     st.write(summary_stats)
#
#     # Plot cumulative returns
#     st.subheader("Cumulative Returns")
#     fig_cumulative_returns = pf.plot_returns(returns)
#     st.write(fig_cumulative_returns)
#
#     # Plot rolling volatility and monthly returns heatmap using the tears module
#     with io.StringIO() as buf, contextlib.redirect_stdout(buf):
#         pf.create_simple_tear_sheet(returns, live_start_date=None, benchmark_rets=None)
#         output = buf.getvalue()
#
#     st.write(output)



def plot_stock_data(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                  open=df['Open'],
                                  high=df['High'],
                                  low=df['Low'],
                                  close=df['Close'],
                                  name='Stock Price'))
    fig.update_layout(title='Stock Price History',
                      xaxis_title='Date',
                      yaxis_title='Price')
    return fig


def plot_backtest_results(df, predictions):
    df_backtest = df.iloc[-len(predictions):].copy()
    df_backtest['Predictions'] = predictions
    df_backtest['Buy_Signal'] = (df_backtest['Predictions'] == 1)

    buy_signals = df_backtest[df_backtest['Buy_Signal']]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_backtest.index,
                                  open=df_backtest['Open'],
                                  high=df_backtest['High'],
                                  low=df_backtest['Low'],
                                  close=df_backtest['Close'],
                                  name='Stock Price'))
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                             mode='markers',
                             marker=dict(color='green', size=8, symbol='circle'),
                             name='Buy Signals'))

    fig.update_layout(title='Backtest Results',
                      xaxis_title='Date',
                      yaxis_title='Price')

    return fig


def download_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df


def predict_stock_price(df, model_choice):
    df['Label'] = df['Close'].shift(-5)
    df.dropna(inplace=True)
    X = df.drop(['Label'], axis=1)
    y = (df['Label'] > df['Close']).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == "RandomForest":
        model = RandomForestClassifier()
    elif model_choice == "KNN":
        model = KNeighborsClassifier()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return predictions, accuracy


def backtest(df, predictions):
    df_backtest = df.iloc[-len(predictions):].copy()
    df_backtest['Predictions'] = predictions
    df_backtest['Buy_Signal'] = (df_backtest['Predictions'] == 1)

    initial_capital = 10000
    capital = initial_capital
    shares = 0

    for idx, row in df_backtest.iterrows():
        if row['Buy_Signal']:
            buy_amount = capital * 0.1
            shares += buy_amount / row['Close']
            capital -= buy_amount

    final_capital = capital + (shares * df_backtest.iloc[-1]['Close'])
    return final_capital


# ------------------------------   Modeling the Returns    ------------------------------#
# ------------------------------ Parameters and Variables  ------------------------------#
# mu = Daily Expected Return: calculated as the mean of the daily returns.
# var = Variance of the data.
# drift = mu - 0.5*var
# desvest = Standar Deviation of the Data.
# n_simulations = desired number of simulations.
# days = the number of days we are going to simulate.
# epsilon = stochastic component generated by the inverse of the normal distribution.
# returns = simulated daily returns.
# returns_interval_1 = interval that is 1.96 standard deviation from its mean.
# returns_interval_2 = interval that is -1.96 standard deviation from its mean.
# expected_returns = expected returns vector.
# S0 = the initial value of the stock
# S = price of the stock at time t
# S_interval_1 = price of the stock 1 year from now of interval 1.
# S_interval_2 = price of the stock 1 year from now of interval 2.

# ------------------------------        Parameters        ------------------------------#
def Monte_carlo(df):
    df_pct = df['Adj Close'].pct_change()
    mu = df_pct.mean()
    var = df_pct.var()
    drift = mu - (0.5 * var)
    desvest = df_pct.std()
    n_simulations = 1000
    days = np.arange(7)
    last_price = df['Adj Close'][-1]

    S0 = last_price

    # ------------------------------         Variables        ------------------------------#

    epsilon = norm.ppf(np.random.rand(len(days), n_simulations))
    returns = drift + desvest * epsilon
    returns_interval_1 = np.zeros(len(days))
    returns_interval_2 = np.zeros(len(days))
    expected_returns = np.zeros(len(days))

    for t in range(1, len(days)):
        returns_interval_1[t] = drift * (t - days[0]) + desvest * 1.96 * np.sqrt(t - days[0])
        returns_interval_2[t] = drift * (t - days[0]) + desvest * -1.96 * np.sqrt(t - days[0])

    S = np.zeros_like(returns)
    S_interval_1 = np.zeros_like(returns_interval_1)
    S_interval_2 = np.zeros_like(returns_interval_2)
    S_interval_1[0] = S0
    S_interval_2[0] = S0
    S[0] = S0
    expected_returns[0] = S0
    # ------------------------------       Price Modeling     ------------------------------#

    for t in range(1, len(days)):
        S[t] = S[t - 1] * np.exp(returns[t])
        S_interval_1[t] = S0 * np.exp(returns_interval_1[t])
        S_interval_2[t] = S0 * np.exp(returns_interval_2[t])
        expected_returns[t] = expected_returns[t - 1] * np.exp(mu)

    return S, S_interval_1, S_interval_2, expected_returns


def plot_monte_carlo(S, S_interval_1, S_interval_2, expected_returns):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.DataFrame(S), color='green', linewidth=1, alpha=0.05)

    fig.patch.set_alpha(0.5)
    fig.patch.set_facecolor('#e6eef5')

    ax.set_facecolor('#e6eef5')
    ax.patch.set_alpha(0.2)

    plt.plot(S_interval_1, label='upper bound')
    plt.plot(S_interval_2, label='lower bound', c='orange')
    plt.plot(expected_returns, c='red', label='expected price', alpha=1)
    plt.title('Price of Athe Rabigh Refining and Petrochemical Company stock 7 days from now: 95% Confidence Interval',
              fontweight='bold', c='black')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    plt.legend(loc='upper left')

    return fig


st.set_page_config(layout="wide")

st.title("Stock Price Predictor")

col1, col2 = st.columns(2)

with st.sidebar:
    ticker = st.text_input("Enter stock ticker:", "2380.SR")
    start_date = st.date_input("Start date:", datetime.now() - timedelta(days=365*10))
    end_date = st.date_input("End date:", datetime.now())
    model_choice = st.selectbox("Select model:", ("RandomForest", "KNN"))

accuracy = None

if st.button("Predict"):
    df = download_stock_data(ticker, start_date, end_date)

    # Monte Carlo Simulation chart
    st.title('Monte Carlo Simulations')
    monte_carlo_chart = Monte_carlo(df)
    st.pyplot(plot_monte_carlo(monte_carlo_chart[0], monte_carlo_chart[1],
                               monte_carlo_chart[2], monte_carlo_chart[3]))
    st.text(f'Upper bound of the price in 7 trading days: {monte_carlo_chart[1][-1]}')
    st.text(f'Lower bound of the price in 7 trading days: {monte_carlo_chart[2][-1]}')


    predictions, accuracy = predict_stock_price(df, model_choice)
    final_capital = backtest(df, predictions)
    returns = generate_returns_series(df, predictions)

    # Stock Price History chart
    stock_chart = plot_stock_data(df)
    col1.plotly_chart(stock_chart, use_container_width=True)

    # Backtest Results chart
    backtest_chart = plot_backtest_results(df, predictions)
    col2.plotly_chart(backtest_chart, use_container_width=True)

    # Display backtest statistics
    display_backtest_statistics(returns)



if accuracy is not None:
    # Model Accuracy and Backtest result
    st.markdown("---")
    st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")
    st.write(f"**Backtest Result:** ${final_capital:.2f}")

