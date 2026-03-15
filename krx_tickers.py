import time
from typing import List, Optional

import pandas as pd
import requests
import yfinance as yf


def _download_krx_table() -> pd.DataFrame:
    """
    KRX에서 전체 상장법인(코스피/코스닥/코넥스) 목록을 다운로드한다.
    lxml이 없어도 동작하도록 requests + pandas.read_html(response.text)를 사용한다.
    """
    url = (
        "http://kind.krx.co.kr/corpgeneral/corpList.do"
        "?method=download&searchType=13"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    if not tables:
        raise RuntimeError("KRX 종목 리스트를 가져오지 못했습니다.")
    df = tables[0]
    return df


def get_krx_yahoo_tickers(
    limit: Optional[int] = 300,
    verify_with_yfinance: bool = True,
) -> List[str]:
    """
    KRX 상장 종목 전체를 가져와 야후 파이낸스 티커 형식으로 변환한다.

    - 코스피: .KS
    - 코스닥/코넥스: .KQ

    verify_with_yfinance=True 이면, yfinance로 실제로
    최근 1개월 데이터를 받아올 수 있는 종목만 남긴다.
    limit가 설정되어 있으면 최대 해당 개수까지만 사용한다.
    """
    df = _download_krx_table()

    # 종목코드 6자리 패딩
    df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)

    # 보통주만 사용 (우선주 등 제외)
    if "주식종류" in df.columns:
        df = df[df["주식종류"] == "보통주"]

    # 시장 구분에 따라 야후 suffix 부여
    def to_yahoo(row: pd.Series) -> str:
        code = row["종목코드"]
        market = str(row.get("시장구분", ""))
        if "코스피" in market:
            suffix = ".KS"
        else:
            # 코스닥/코넥스 등은 대부분 .KQ
            suffix = ".KQ"
        return f"{code}{suffix}"

    df["yahoo_ticker"] = df.apply(to_yahoo, axis=1)
    tickers = df["yahoo_ticker"].drop_duplicates().tolist()

    if limit is not None:
        tickers = tickers[:limit]

    if not verify_with_yfinance:
        return tickers

    # 실제로 데이터 다운로드가 되는 종목만 남긴다.
    valid_tickers: List[str] = []
    print(f"KRX 종목 {len(tickers)}개 중 실제로 데이터가 있는 종목 필터링 중...")

    for i, t in enumerate(tickers, start=1):
        try:
            df_test = yf.download(t, period="1mo", interval="1d", progress=False)
            if not df_test.empty:
                valid_tickers.append(t)
        except Exception:
            # 실패하는 티커는 그냥 건너뜀
            pass

        # 너무 빠른 요청 방지
        time.sleep(0.1)

        if i % 20 == 0:
            print(f" - {i}개 검사 완료, 유효 티커 {len(valid_tickers)}개")

    print(f"최종 사용 티커 수: {len(valid_tickers)}개")
    return valid_tickers

