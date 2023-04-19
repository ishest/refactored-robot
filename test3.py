import streamlit as st
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import LabelEncoder


import numpy as np
from scipy.stats import norm
import pickle
from xgboost import XGBClassifier

# metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve, RocCurveDisplay

# import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# import boruta
from boruta import BorutaPy

# warnings
import warnings
warnings.filterwarnings('ignore')

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
import io
import contextlib

# sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve, RocCurveDisplay

# import classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier

from scipy.stats import norm
from scipy.optimize import minimize
import statsmodels.api as sm
# from prophet import Prophet
# from prophet.plot import plot_plotly, add_changepoints_to_plot
#

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit


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


def data_processing():
    df = yf.download('2380.SR')
    df_pct = df[['Adj Close']].pct_change().dropna()
    df.ta.strategy('All')
    df = df.drop(['DPO_20', 'ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26', 'INC_1'], axis=1)

    data = df.copy()

#     n = 7
    # define target (label)
    pct_change = data['Close'].pct_change(n)
    data['predict'] = np.where(pct_change.shift(-n) > 0, 1, 0)

    # drop unwanted columns
    data = data.drop(['HILOl_13_21', 'HILOs_13_21', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2',
                      'QQEl_14_5_4.236', 'QQEs_14_5_4.236', 'SUPERTl_7_3.0', 'SUPERTs_7_3.0',
                      'Open', 'High', 'Low', 'Volume'], axis=1)

    data = data[200:]

    # backfill columns to address missing values
    data = data.bfill(axis=1)
    data = data[:-n]  # to take care of n-days ahead prediction

    c = data['predict'].value_counts()

    # class weight function
    def cwts(dfs):
        c0, c1 = np.bincount(dfs['predict'])
        w0 = (1 / c0) * (len(df)) / 2
        w1 = (1 / c1) * (len(df)) / 2
        return {0: w0, 1: w1}

    # check class weights
    class_weight = cwts(data)

    # With the calculated weights, both classes gain equal weight
    class_weight[0] * c[0], class_weight[1] * c[1]



    X = data.drop('predict', axis=1)
    feature_names = X.columns

    features = pd.read_csv('feature_list.csv')['Feature'].to_list()
    X = X[features]

    y = data['predict'].values
    # pandas-ta converts all dtype to objects
    y = y.astype(int)

    # Always keep shuffle = False for financial time series
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # convert to array
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    # # perform normalization
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(X_train)
    scaled_test = scaler.transform(X_test)





    # # # define random forest classifier
    # # forest = RandomForestClassifier(n_jobs=-1,
    # #                                 class_weight=cwts(data),
    # #                                 random_state=42,
    # #                                 max_depth=5)
    # #
    # # # train the model
    # # forest.fit(scaled_train, y_train)
    # #
    # # # define Boruta feature selection method
    # # feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=0)
    # #
    # # # find all relevant features
    # # # takes input in array format not as dataframe
    # # feat_selector.fit(scaled_train, y_train)
    # #
    # # # call transform() on X to filter it down to selected features
    # # X_filtered = feat_selector.transform(scaled_train)
    # #
    # # # zip my names, ranks, and decisions in a single iterable
    # # feature_ranks = list(zip(feature_names,
    # #                          feat_selector.ranking_,
    # #                          feat_selector.support_))
    # #
    # # # iterate through and print out the results
    # # for feat in feature_ranks:
    # #     print(f'Feature: {feat[0]:<30} Rank: {feat[1]:<5} Keep: {feat[2]}')
    # #
    # # selected_rf_features = pd.DataFrame({'Feature': feature_names,
    # #                                      'Ranking': feat_selector.ranking_})
    # #
    # # # selected_rf_features#.sort_values(by='Ranking')
    # #
    # # selected_rf_features[selected_rf_features['Ranking'] == 1]
    #
    # X_test_filtered = feat_selector.transform(scaled_test)

    return scaled_train, scaled_test, y_train, y_test


def predict_stock_price(df, data_processing, model_choice):
    # df['Label'] = df['Close'].shift(-5)
    # df.dropna(inplace=True)
    # X = df.drop(['Label'], axis=1)
    # y = (df['Label'] > df['Close']).astype(int)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == "RandomForest":
        # Load the model
        with open('models/forest.pkl', 'rb') as f:
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
    predictions = model.predict(data_processing[1])
    accuracy = accuracy_score(data_processing[3], predictions)

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
            buy_amount = capital * 1
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
    start_date = st.date_input("Start date:", datetime.now() - timedelta(days=365*10))
    end_date = st.date_input("End date:", datetime.now())
    model_choice = st.selectbox("Select model:", ("RandomForest", "KNN", "SVM", "logReg",
                                                  # "XGboost", "Boosting",
                                                  "Decision Tree",))
                                                  # "stacking",
                                                  # "stacking1", "stacking2", "voting"))

accuracy = None

if st.button("Predict"):
    # print(pd.read_csv('feature_list.csv')['Feature'].to_list())
    df = download_stock_data(ticker, start_date, end_date)
    print(df)
    data_process = data_processing()

    # Monte Carlo Simulation chart
    st.title('Monte Carlo Simulations')
    monte_carlo_chart = Monte_carlo(df)
    st.pyplot(plot_monte_carlo(monte_carlo_chart[0], monte_carlo_chart[1],
                               monte_carlo_chart[2], monte_carlo_chart[3]))
    st.text(f'Upper bound of the price in 7 trading days: {monte_carlo_chart[1][-1]}')
    st.text(f'Lower bound of the price in 7 trading days: {monte_carlo_chart[2][-1]}')


    predictions, accuracy = predict_stock_price(df, data_process, model_choice)
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
    st.write(f"**Model Signal is:** {predictions[-1]}")

