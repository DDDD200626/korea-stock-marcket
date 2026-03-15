from typing import Optional

import pandas as pd
import yfinance as yf


def load_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    야후 파이낸스에서 주가 데이터를 다운로드한다.

    :param ticker: 예) "005930.KS" (삼성전자)
    :param period: 예) "1y", "5y", "max"
    :param interval: 예) "1d", "1h"
    :return: 시가/고가/저가/종가/거래량 등이 포함된 DataFrame
    """
    df: pd.DataFrame = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

    if df.empty:
        raise ValueError(f"티커 {ticker} 에 대한 데이터를 가져오지 못했습니다.")

    # 결측치 제거
    df = df.dropna()
    return df

