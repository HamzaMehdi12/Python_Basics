import pandas as pd
import time
import yfinance as yf
import matplotlib.pyplot as plt

from pandas_mastery_1 import LabelEncoding, Imputer
from scipy import stats

#For dashboard
import streamlit as st
import seaborn as sns


class Stockdata:
    def __init__(self):
        "Stock data with API and dashboard creation"

        ticker = "NVDA"
        interval = "5m"
        period = "1mo"

        #Downloading data
        # intervals can be '1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo'
        # period can be: '1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', etc.
        print(f"Downloading {ticker} data {interval}, {period}...")
        self.df = yf.download(ticker, period=period, interval=interval, group_by='ticker', progress=False)

        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = [col[1] for col in self.df.columns]

        self.df.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj_Close',
            'Volume': 'Volume'
        }, inplace=True)

        print(f"The dataframe for stocks is as follows:")
        print(self.df.head())
        self.df.info()

        self.df.to_csv("OPEN_raw_data.csv")
        print("File saved to path")

        self.preprocessing()
        self.Analytical_Modeling()
        self.Dashboard_Creation()

    def SMA(self, df, window = 0):
        "Simple Moving Average"
        return df['Close'].rolling(window).mean()
    
    def RSI(self, df, period = 14, ema=True):
        "Relative Strength Indicator"
        close_delta = df['Close'].diff()

        #Making 2 series
        up = close_delta.clip(lower=0) #gain
        down = -1 * close_delta.clip(upper = 0) #loss

        if ema == True:
            #Using exponential moving averages
            ma_avg = up.ewm(com = period - 1, adjust = True, min_periods = period).mean() #Moving Average gain
            ma_loss = down.ewm(com = period - 1, adjust = True, min_periods = period).mean() #Moving Average Loss
        else:
            #Using simple moving average
            ma_avg = up.rolling(window = period, adjust = False).mean() #Moving Average gain
            ma_loss = down.rolling(window = period, adjust = False).mean() #Moving Average Loss

        rsi = ma_avg / ma_loss

        rsi = 100 - (100 / (1 + rsi))

        return rsi
    
    def ROC(self, df, window = 14):
        "Calculates the return on capital"
        return df['Close'].pct_change(window).mean() * 100
    
    def Bollinger_Bands(self, df, window = 20, n_std = 2):
        "Calculates the strength of a stock"
        SMA = self.SMA(df, window)
        std = df['Close'].rolling(window).std()
        up = SMA + (std * n_std)
        down = SMA - (std * n_std)

        return up, down
    
    def Rolling_Standard_Deviation(self, df, window = 14):
        "Standard Deviation"
        return df['Close'].rolling(window).std()

    def preprocessing(self):
        "Now we preprocess the data like we did in our last file"
        #Feature Engineering
        # -> Datetime
        print("Applying Datetime Engineering")
        self.df.index = pd.to_datetime(self.df.index)
        self.df.index = self.df.index.tz_localize(None)
        full_index = pd.date_range(self.df.index.min(), self.df.index.max(), freq='H')
        self.df = self.df.reindex(full_index)
        self.df[['Open','High','Low','Close','Volume']] = self.df[['Open','High','Low','Close','Volume']].fillna(method='ffill')
        print("Completed Datetime formatting!")

        print("Adding features like Returns and mean values at the end of the dataframe for 7 and 14 days")
        self.df['Returns'] = self.df['Close'].pct_change().mean()#Calculates the returns based on the column Close
        self.df['SMA_7'] = self.SMA(self.df, window = 7) #Calculates the 1 week mean rolling price of the stock
        self.df['RSI_7'] = self.RSI(self.df, period=7) #Calculates RSI for 1 week
        self.df['ROC_7'] = self.ROC(self.df, window = 7)#Calculates ROC for 1 week mean
        self.df['BB_Up_7'], self.df['BB_Low_7'] = self.Bollinger_Bands(self.df, window = 7) #Calculates for a week
        self.df['Std_7'] = self.Rolling_Standard_Deviation(self.df, window = 7) #Standard Deviation for a week

        self.df['SMA_14'] = self.SMA(self.df, window = 7) #Calculates the 1 week mean rolling price of the stock
        self.df['RSI_14'] = self.RSI(self.df, period=7) #Calculates RSI for 1 week
        self.df['ROC_14'] = self.ROC(self.df, window = 7)#Calculates ROC for 1 week mean
        self.df['BB_Up_14'], self.df['BB_Low_14'] = self.Bollinger_Bands(self.df, window = 7) #Calculates for a week
        self.df['Std_14'] = self.Rolling_Standard_Deviation(self.df, window = 7) #Standard Deviation for a week
        print("Added the values completely")

        #Checking for Nan values all around the file and replacing it with 0
        if self.df.isna().any().any():
            print("Nan values detected")
            self.df = self.df.fillna(0)
            print("Completed filling and removing Nan values")
        else:
            print("No values found")

        
        print(f"Now the dataset is as follows: \n")
        time.sleep(2)
        with pd.option_context('display.max_columns', None):
            print(self.df.head(5).to_string())#to_string for whole line output without line wrapping

        return self
    
    #Plotting for EDA
    def Analytical_Modeling(self):
        "The preprocessing is completed, now going for the time series"
        #Plotting the results for time series analysis
        #EDA plotting
        print('Plotting the EDA analysis')

        fig, axes = plt.subplots(3, 2, figsize=(16,12))
        fig.suptitle("Exploratory Data Analysi (EDA) for Stock Data", fontsize = 16)

        #SMA_7 & SMA_14
        axes[0, 0].plot(self.df['Close'], label = "Close Price", alpha = 0.7) #alpha is transparency
        axes[0, 0].plot(self.df['SMA_7'], label = "7-Day SMA", color = "orange")
        axes[0, 0].plot(self.df['SMA_14'], label = "14-Day SMA", color = "red")
        axes[0, 0].set_title("Rolling Avergae 7 and 14 days")
        axes[0, 0].legend()

        #RSI_7 and RSI_14
        axes[0, 1].plot(self.df['Close'], label = "Close Price", alpha = 0.7)
        axes[0, 1].plot(self.df['RSI_7'], label = "7-Day RSI", linestyle = "--", color = "red")
        axes[0, 1].plot(self.df['RSI_14'], label = "14-Day RSI", linestyle = "--", color = "green")
        axes[0, 1].set_title("RSI 7 and 14 days")
        axes[0, 1].legend()

        #Bollinger Bands
        axes[1, 0].plot(self.df['Close'], label = "Close Price", alpha = 0.7)
        axes[1, 0].plot(self.df['BB_Up_7'], label = "7-Day BB", linestyle = "--", color = "purple")
        axes[1, 0].plot(self.df['BB_Up_14'], label = "14-Day BB", linestyle = "--", color = "gray")
        axes[1, 0].fill_between(self.df.index, self.df['BB_Low_7'], self.df['BB_Up_7'], color = "gray", alpha = 0.2)
        axes[1, 0].set_title("Bollinger Bands 7 days")
        axes[1, 0].legend()

        #ROC
        axes[1, 1].plot(self.df['Close'], label = "Close Price", alpha = 0.7)
        axes[1, 1].plot(self.df['ROC_7'], label = "7-Day ROC", linestyle = "-.", color = "red")
        axes[1, 1].plot(self.df['ROC_14'], label = "14-Day ROC", linestyle = "-.", color = "green")
        axes[1, 1].set_title("ROC 7 and 14 days")
        axes[1, 1].legend()

        #Volatility
        axes[2, 0].plot(self.df['Close'], label = "Close Price", alpha = 0.7)
        axes[2, 0].plot(self.df['Std_7'], label = "7-Day Std", linestyle = ":", color = "brown")
        axes[2, 0].plot(self.df['Std_14'], label = "14-Day Std", linestyle = ":", color = "pink")
        axes[2, 0].set_title("Standard Deviation 7 and 14 days")
        axes[2, 0].legend()


        #Histogram of Returns
        axes[2, 1].hist(self.df['Returns'].dropna(), bins = 50, color = "blue", alpha = 0.7, edgecolor = "black")
        axes[2, 1].set_title("Returns Distribution")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        print("EDA Plotted")

        #Detecting patterns from engineered quantities
        anomalies = self.df[(self.df['RSI_7'] > 50) | (self.df['RSI_7'] < 15) ]#Overbought vs Oversold
        print(f"Anomalities observed: {anomalies} and are {len(anomalies)} many")

        return self
    
    def Dashboard_Creation(self):
        "Finally, creating the dashboard for the model"

        print("Saving the results in a files")
        print("1. Raw prices -> prices.csv, would contain the following")

        prices = self.df[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
        print(prices.columns)
        prices.to_csv("prices.csv")
        print("Saved raw prices in prices.csv")

        print("2. Indicators -> indicators.csv, would have the following")
        indicators = self.df[['Close', 'SMA_7', 'SMA_14', 'RSI_7', 'RSI_14', 'ROC_7', 'ROC_14', 'BB_Low_7', 'BB_Up_7', 'BB_Low_14', 'BB_Up_14', 'Std_7', 'Std_14']]
        print(indicators.columns)
        indicators.to_csv("indicators.csv")
        print("Saved indicators to indicators.csv")

        print("3. Summary -> summary.csv, would have the following")
        summary = pd.DataFrame( {
            "Metrics": ["Mean Return", "Volatility", "Max_Close", "Min_Close"],
            "Value": [
                self.df["Returns"].mean(),
                self.df["Returns"].std(),
                self.df["Close"].max(),
                self.df["Close"].min()
            ]
        })
        print(summary.columns)
        summary.to_csv("summary.csv")
        print("Saved summary to summary.csv")

        #Title dashboard
        st.title("📊 Stock Dashboard")

        #Line Charts with SMA
        st.subheader("Prices with Moving Avg")
        fig, ax = plt.subplots(figsize = (10, 5))
        ax.plot(prices['Close'], label = "Close")
        ax.plot(indicators['SMA_7'], label = "SMA 7", color = "orange")
        ax.plot(indicators['SMA_14'], label = "SMA 14", color = "blue")
        ax.legend()
        st.pyplot(fig)

        
        st.subheader("RSI")
        fig, ax = plt.subplots(figsize = (10, 5))
        ax.plot(prices['Close'], label = "Close")
        ax.plot(indicators['RSI_7'], label = "RSI 7", linestyle = "-", color = "green")
        ax.plot(indicators['RSI_14'], label = "RSI 14", linestyle = "-", color = "blue")
        ax.legend()
        st.pyplot(fig)


        st.subheader("ROC")
        fig, ax = plt.subplots(figsize = (10, 5))
        ax.plot(prices['Close'], label = "Close")
        ax.plot(indicators['ROC_7'], label = "ROC 7", linestyle = "--", color = "red")
        ax.plot(indicators['ROC_14'], label = "ROC 14", linestyle = "--", color = "orange")
        ax.legend()
        st.pyplot(fig)


        st.subheader("Volatility")
        fig, ax = plt.subplots(figsize = (10, 5))
        ax.plot(prices['Close'], label = "Close")
        ax.plot(indicators['Std_7'], label = "Std 7", linestyle = ":", color = "green")
        ax.plot(indicators['Std_7'], label = "Std 7", linestyle = ":", color = "blue")
        ax.legend()
        st.pyplot(fig)


        st.subheader("Bollinger Bands")
        fig, ax = plt.subplots(figsize = (10, 5))
        ax.plot(prices['Close'], label = "Close")
        ax.plot(indicators['BB_Low_7'], label = "BB_Low 7", linestyle = "dashed", color = "black")
        ax.plot(indicators['BB_Up_7'], label = "BB_Up 7", linestyle = "dashed", color = "gray")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Returns Distribution")
        fig, ax = plt.subplots(figsize = (10, 5))
        ax.hist(prices['Returns'].dropna(), bins = 100, color = "blue", alpha = 0.7, edgecolor = "black")
        st.pyplot(fig)

        st.subheader("Correlational Heatmap")
        pivot = self.df[['Open','High','Low','Close','Volume','SMA_7','SMA_14','RSI_7', 'RSI_14', 'ROC_7', 'ROC_14' ,'Std_7', 'Std_14', 'BB_Low_7', 'BB_Up_7', 'BB_Low_14', 'BB_Up_14']].fillna(0) #If group values then use pivot_table
        corr = pivot.pct_change().corr()
        fig, ax = plt.subplots(figsize = (10, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("📌 Key Metrics")

        mean_return = prices['Returns'].mean()
        volatility = prices['Returns'].std()
        sharpe = mean_return / volatility if volatility != 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Return", f"{mean_return:.2%}")
        col2.metric("Volatility", f"{volatility:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")


if __name__ == "__main__":
    print("Starting our Fianncial Data Analyzer")
    Stocks = Stockdata()
