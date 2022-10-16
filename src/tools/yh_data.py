import os
import pandas as pd
from tqdm import tqdm
from yahoo_fin.stock_info import tickers_sp500
from yahoo_fin.stock_info import get_data

def download_sp500(start_date="1999-12-01", end_date="2020-12-31", itvl='1d'):
    """
    Download SP500 historical price information

    Args:
        start_date (str, optional): The date the price history should begin. Defaults to "2000-01-01".
        end_date (str, optional): The date the price history should end. Defaults to "2020-12-31".
        itvl (str, optional): could be `1d`, `1mo`, `1wk`. Defaults to '1d'.
    """    
    tickers = tickers_sp500()
    # data_folder ="./yahoo_sp500_data/"
    err_tickers = []
    data = []
    # if not os.path.exists(data_folder):
    #     os.makedirs(data_folder)

    for ticker in tqdm(tickers):
        # print(f"Download historical data for {ticker}")
        try:
            df = get_data(ticker, start_date=start_date, end_date=end_date, interval=itvl, index_as_date=False)
            data.append(df)
        except:
            # print(f'Fail to download {ticker}')
            err_tickers.append(ticker)
    print(f'Finish!\nError tickers failed to download: {err_tickers}')
    df = pd.concat(data, ignore_index=True)
    # `adjclose` will be the price we need to calculate return
    return df[['date', 'ticker', 'adjclose']]

def download_spy(start_date="2000-01-01", end_date="2020-12-31", itvl='1d'):
    df = get_data('spy', start_date=start_date, end_date=end_date, interval=itvl, index_as_date=False)
    return df[['date', 'ticker', 'adjclose']]

if __name__ == '__main__':
    df = download_sp500("1999-12-01", "2022-03-31")
    df.to_csv('./sp500_daily_prc.csv', index=False)
    df = download_spy("1999-12-01", "2022-03-31")
    df.to_csv('./spy_daily_prc.csv', index=False)