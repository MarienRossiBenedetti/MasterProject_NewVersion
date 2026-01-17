import yfinance as yf
import pandas as pd

def import_yahoo(ticker: str, start: str, end: str, raw: bool) -> pd.DataFrame:
    df = yf.download(tickers=ticker, start=start, end=end, auto_adjust=False)
    df.attrs['name'] = ticker

    if raw == False:
        df = df['Adj Close']
        df.columns = ['adj close']

    print(f"✅ {ticker} data successfully imported")
    return df