import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    기본적인 기술적 지표와 타깃 컬럼을 추가한다.
    - return: 일간 수익률
    - ma5, ma20: 5일, 20일 이동평균
    - vol_ma5: 5일 평균 거래량
    - ma_diff: 장단기 이동평균 차이 (ma5 - ma20)
    - rsi14: 14일 RSI
    - macd, macd_signal, macd_hist: MACD 관련 지표
    - bb_upper, bb_lower, bb_width: 볼린저 밴드 상단/하단/폭
    - target: 다음 날 종가가 오늘보다 높으면 1, 아니면 0
    """
    df_feat = df.copy()

    close = df_feat["Close"]

    # 일간 수익률
    df_feat["return"] = close.pct_change()

    # 이동평균
    df_feat["ma5"] = close.rolling(window=5).mean()
    df_feat["ma20"] = close.rolling(window=20).mean()

    # 장단기 이동평균 차이
    df_feat["ma_diff"] = df_feat["ma5"] - df_feat["ma20"]

    # 거래량 이동 평균
    df_feat["vol_ma5"] = df_feat["Volume"].rolling(window=5).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(window=14).mean()
    roll_down = loss.rolling(window=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df_feat["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df_feat["macd"] = macd
    df_feat["macd_signal"] = macd_signal
    df_feat["macd_hist"] = macd - macd_signal

    # 볼린저 밴드 (20일, 2표준편차)
    ma20 = df_feat["ma20"]
    std20 = close.rolling(window=20).std()

    # pandas 3.x 호환: 인덱스를 명시적으로 정렬/정렬해 Series 로 강제
    bb_upper = (ma20 + 2 * std20).astype("float64")
    bb_lower = (ma20 - 2 * std20).astype("float64")

    # 길이나 인덱스가 맞지 않아도 안전하게 할당되도록 reindex
    bb_upper = bb_upper.reindex(df_feat.index)
    bb_lower = bb_lower.reindex(df_feat.index)

    df_feat["bb_upper"] = bb_upper
    df_feat["bb_lower"] = bb_lower
    df_feat["bb_width"] = (df_feat["bb_upper"] - df_feat["bb_lower"]) / (ma20.astype("float64") + 1e-9)

    # 타깃: 다음 날 종가가 오늘보다 크면 1
    df_feat["target"] = (close.shift(-1) > close).astype(int)

    # 롤링/shift로 생긴 결측치 제거
    df_feat = df_feat.dropna()

    return df_feat

