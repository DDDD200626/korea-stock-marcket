import pandas as pd

from data_loader import load_data
from features import add_features
from krx_tickers import get_krx_yahoo_tickers
from model import train_model, predict_next_day


def main() -> None:
    # KRX 전체 상장 종목에서 야후 파이낸스 티커 자동 생성
    print("[0/4] KRX 전체 종목 리스트에서 티커 생성 중...")
    tickers = get_krx_yahoo_tickers(limit=300, verify_with_yfinance=True)

    if not tickers:
        raise RuntimeError("사용 가능한 한국 주식 티커를 찾지 못했습니다.")

    period = "5y"

    all_feat_list: list[pd.DataFrame] = []

    print(f"[1/4] 여러 종목 데이터 다운로드 및 피처 생성 (총 {len(tickers)}개)")
    for ticker in tickers:
        print(f" - {ticker} 다운로드 중...")
        df_raw = load_data(ticker=ticker, period=period)
        df_feat = add_features(df_raw)
        df_feat["ticker"] = ticker
        all_feat_list.append(df_feat)

    # 여러 종목 데이터를 하나의 DataFrame으로 합치고, 날짜 순으로 정렬
    df_all = pd.concat(all_feat_list, axis=0)
    df_all = df_all.sort_index()

    print(f"[2/4] 총 샘플 수: {len(df_all)}")

    print(f"[3/4] 모델 학습 및 평가")
    model, feature_cols = train_model(df_all)

    print(f"[4/4] 내일 방향 예측 (마지막 종목 기준)")
    last_ticker = tickers[-1]
    df_last = df_all[df_all["ticker"] == last_ticker]
    predict_next_day(model, feature_cols, df_last, ticker=last_ticker)


if __name__ == "__main__":
    main()

