import streamlit as st
import pandas as pd
from arch import arch_model
import ta


import numpy as np
import pickle


# warnings
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pyfolio as pf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


# metrics
from sklearn.metrics import accuracy_score
from scipy.stats import norm

from sklearn.model_selection import train_test_split


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
    st.write(summary_stats.loc[['Sharpe ratio', 'Annual volatility', 'Max drawdown', 'Annual return']])




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


def data_processing(data):
    def create_features(data, target_day=7):
        data['Target'] = data['Close'].shift(-target_day) > data['Close']
        data.dropna(inplace=True)
        return data

    def add_moving_averages(data, windows=[5, 10, 30]):
        for window in windows:
            data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
            data[f'EMA_{window}'] = data['Close'].ewm(span=window).mean()
        return data

    def add_rsi(data, window=14):
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window).rsi()
        return data

    def add_macd(data, short_window=12, long_window=26, signal_window=9):
        macd_indicator = ta.trend.MACD(data['Close'], short_window, long_window, signal_window)
        data['MACD'] = macd_indicator.macd()
        data['MACD_signal'] = macd_indicator.macd_signal()
        data['MACD_diff'] = macd_indicator.macd_diff()
        return data

    def create_features(data, target_day=7):
        data = add_moving_averages(data)
        data = add_rsi(data)
        data = add_macd(data)
        data['Target'] = data['Close'].shift(-target_day) > data['Close']
        data.dropna(inplace=True)
        return data

    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)

    data = create_features(data)

    return data


def predict_stock_price(data_processing, model_choice):

    if model_choice == "RandomForest":
        # Load the model
        with open('models/stochy_ai_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_choice == "KNN":
        with open('models/knn.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_choice == "Boosting":
        with open('models/boosting.pkl', 'rb') as f:
            model = pickle.load(f)
            model.fit(data_processing[0], data_processing[2])
    elif model_choice == "XGboost":
        with open('models/gboost.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_choice == "logReg":
        with open('models/logReg.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_choice == "Decision Tree":
        with open('models/decision_tree.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_choice == "stacking":
        with open('models/stacking.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_choice == "stacking1":
        with open('models/stacking1.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_choice == "stacking2":
        with open('models/stacking2.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_choice == "voting":
        with open('models/voting.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_choice == "SVM":
        with open('models/svm.pkl', 'rb') as f:
            model = pickle.load(f)


    # model.fit(data_processing, y_train)
    predictions = model.predict(data_processing.drop(['Date', 'Close', 'Target'], axis=1))
    accuracy = accuracy_score(data_processing['Target'], predictions)

    preds = model.predict(data_processing.iloc[-1:].drop(['Date', 'Close', 'Target'], axis=1))

    return predictions, accuracy, preds


predictions = []


def backtest(df, predictions):
    df_backtest = df.iloc[-len(predictions):].copy()
    df_backtest['Predictions'] = predictions
    df_backtest['Buy_Signal'] = (df_backtest['Predictions'] == 1)

    initial_capital = 10000
    capital = initial_capital
    shares = 0

    for idx, row in df_backtest.iterrows():
        if row['Buy_Signal']:
            buy_amount = capital * 1
            shares += buy_amount / row['Close']
            capital -= buy_amount

    final_capital = capital + (shares * df_backtest.iloc[-1]['Close'])
    return final_capital



# ------------------------------        Parameters        ------------------------------#
def Monte_carlo(df):
    df_pct = df['Adj Close'].pct_change()
    mu = df_pct.mean()
    var = df_pct.var()
    drift = mu - (0.5 * var)
    desvest = df_pct.std()
    n_simulations = 1000
    days = np.arange(7)
    # print(df)
    last_price = df['Adj Close'][-1]
    print('last_price', last_price)

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
    plt.title('Price of Rabigh Refining and Petrochemical Company stock 7 days from now: 95% Confidence Interval',
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
    start_date = st.date_input("Start date:", datetime.now() - timedelta(days=365*12))
    end_date = st.date_input("End date:", datetime.now())
    model_choice = st.selectbox("Select model:", ("RandomForest",
                                                  # "KNN", "SVM", "logReg",
                                                  # "XGboost", "Boosting",
                                                  # "Decision Tree",
                                                  ))
                                                  # "stacking",
                                                  # "stacking1", "stacking2", "voting"))

accuracy = None


def garch(df):
    df = df['Adj Close']

    # Step 2: Install the necessary libraries
    # !pip install yfinance
    # !pip install arch

    # Step 3: Prepare the data
    df = 100 * df.pct_change().dropna()

    # Step 4: Fit the eGARCH model
    model = arch_model(df, vol='EGARCH', p=1, q=1)
    results = model.fit()

    # Step 5: Predict the 7-day volatility
    forecast = results.forecast(horizon=7, method='simulation', simulations=1000)
    mean_volatility = np.mean(forecast.simulations.residual_variances[-1]) ** 0.5
    print("Predicted 7-day volatility:", mean_volatility)

    return mean_volatility


if st.button("Predict"):
    # print(pd.read_csv('feature_list.csv')['Feature'].to_list())
    df = download_stock_data(ticker, start_date, end_date)
    print(df)
    print(df['Adj Close'][-1])
    df_ = df.copy()
    data_process = data_processing(df_)
    # print(df)
    # print(data_process)

    # Monte Carlo Simulation chart
    st.title('Monte Carlo Simulations')
    monte_carlo_chart = Monte_carlo(df)
    st.pyplot(plot_monte_carlo(monte_carlo_chart[0], monte_carlo_chart[1],
                               monte_carlo_chart[2], monte_carlo_chart[3]))
    st.text(f'Upper bound of the price in 7 trading days: {round(monte_carlo_chart[1][-1],2)}')
    st.text(f'Lower bound of the price in 7 trading days: {round(monte_carlo_chart[2][-1], 2)}')
#
#
    predictions, accuracy, preds = predict_stock_price(data_process, model_choice)
#     # final_capital = backtest(df, predictions)
    returns = generate_returns_series(df, predictions)
#
#
    # Stock Price History chart
    stock_chart = plot_stock_data(df)
    col1.plotly_chart(stock_chart, use_container_width=True)

    # Backtest Results chart
    backtest_chart = plot_backtest_results(df, predictions)
    col2.plotly_chart(backtest_chart, use_container_width=True)

    # Display backtest statistics
    display_backtest_statistics(returns)

    garch_ = garch((df))
    last_price = df['Adj Close'][-1]
    if preds == 1:
        pred_price = last_price*(1+garch_/100)
        signal = 'BUY'
    else:
        pred_price = last_price*(1-garch_/100)
        signal = 'SELL or do nothing'

    # print(predictions)
    print(pd.DataFrame(preds).value_counts())
    print(pd.DataFrame(predictions).value_counts())



if accuracy is not None:
    # Model Accuracy and Backtest result
    st.title('Forecast info')
    st.markdown("---")
    st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")
    st.write(f"**Model Signal is:** {signal}")
    st.write(f"**Predicted Volatility using eGARCH Method is:** {round(garch_, 2)}%")
    st.write(f"**The current price is:** {round(last_price,2)}")
    st.write(f"**Predicted Price in 7 trading days is:** {round(pred_price, 2)}")
    st.write("DISCLAIMER: The content provided on this website is for informational and "
             "analytical purposes only and should not be construed as investment advice, "
             "a recommendation, or an endorsement of any particular investment strategy, "
             "security, or financial product. The analyses, opinions, and views expressed "
             "on this website are solely those of the authors and contributors and do not "
             "necessarily reflect the views of any third-party organizations or financial institutions."
             "Investing in financial markets carries risks, and past performance does not "
             "guarantee future results. Before making any investment decision, you should "
             "consult with a qualified financial advisor, perform your own research and analysis, "
             "and consider your risk tolerance, financial objectives, and individual circumstances."
             "By using the information on this website, you acknowledge and agree that neither the "
             "authors, contributors, nor the website owner can be held responsible or liable for any "
             "losses, damages, or negative consequences arising from the use of the information provided "
             "or any reliance on the accuracy, completeness, or timeliness of the content.")

    # st.write(f"**Backtest Result:** ${final_capital:.2f}")

