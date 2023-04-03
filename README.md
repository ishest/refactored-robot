# refactored-robot
Stock Price Predictor is a web application that allows users to predict stock prices using machine learning models such as Random Forest and K-Nearest Neighbors (KNN). Users input a stock ticker and date range, and the application retrieves historical stock data using the yfinance library. The application then trains a chosen machine learning model on the historical data and predicts future stock prices.

The main features of the application include:

Displaying stock price history in a candlestick chart using Plotly's graph_objects library.
Backtesting the chosen machine learning model by generating buy signals and calculating final capital based on the predictions.
Displaying the backtest results in a chart with buy signals.
Calculating and displaying various backtest statistics using the Pyfolio library, such as cumulative returns, rolling volatility, and a monthly returns heatmap.
Performing a Monte Carlo simulation to estimate the stock price 7 days in the future and visualizing the results in a chart.
The application also provides model accuracy and backtest results in terms of final capital. The web interface is built using the Streamlit library, which allows for easy and interactive user input and visualization.
