import streamlit as st
import pandas as pd

from data_loader import load_data
from krx_tickers import get_krx_yahoo_tickers
from features import add_features
from model import train_model, predict_next_day


def run_app() -> None:
    st.set_page_config(page_title="한국 주식 예측 데모", layout="wide")

    st.title("📈 한국 주식 방향 예측 데모")

    st.sidebar.header("설정")

    period = st.sidebar.selectbox("기간", ["1y", "3y", "5y"], index=2)

    st.sidebar.markdown("---")
    mode = st.sidebar.radio("예측 대상", ["단일 종목", "KRX 여러 종목"], index=0)

    default_ticker = "005930.KS"
    ticker = st.sidebar.text_input("티커 (야후 형식)", value=default_ticker)

    if st.sidebar.button("예측하기"):
        with st.spinner("데이터 다운로드 및 모델 학습 중..."):
            if mode == "단일 종목":
                tickers = [ticker]
            else:
                tickers = get_krx_yahoo_tickers(limit=50, verify_with_yfinance=False)

            results = []
            for t in tickers:
                try:
                    df_raw: pd.DataFrame = load_data(ticker=t, period=period)
                    if df_raw.empty:
                        continue
                    df_feat: pd.DataFrame = add_features(df_raw)
                    model, feature_cols = train_model(df_feat)
                    results.append((t, model, feature_cols, df_feat))
                except Exception:
                    # 개별 종목 실패는 전체 실행을 막지 않음
                    continue

        if not results:
            st.error("예측에 사용할 수 있는 종목이 없습니다.")
            return

        st.success(f"모델 학습 완료! (총 {len(results)}개 종목)")

        st.subheader("내일 방향 예측")
        for t, model, feature_cols, df_feat in results:
            try:
                st.write(f"티커: **{t}**")
                predict_next_day(model, feature_cols, df_feat, ticker=t)
            except Exception:
                st.write(f"{t}: 예측 실패")

        st.subheader("가격 및 이동평균")
        price_cols = ["Close", "ma5", "ma20"]
        available_cols = [c for c in price_cols if c in df_feat.columns]
        if available_cols:
            data_price = df_feat[available_cols].reset_index(drop=True)
            try:
                st.line_chart(data_price)
            except Exception:
                # 클라우드 환경(pandas/streamlit 버전 차이)에서 차트 그리기가 실패하면
                # 앱이 죽지 않도록 표 형태로만 표시한다.
                st.info("차트를 그리는 중 오류가 발생하여 표 형태로 대신 표시합니다.")
                st.dataframe(data_price)

        if "rsi14" in df_feat.columns:
            st.subheader("RSI(14)")
            data_rsi = df_feat[["rsi14"]].reset_index(drop=True)
            try:
                st.line_chart(data_rsi)
            except Exception:
                st.info("RSI 차트를 그리는 중 오류가 발생하여 표 형태로 대신 표시합니다.")
                st.dataframe(data_rsi)

    else:
        st.info("왼쪽 사이드바에서 티커와 기간을 선택한 뒤 **예측하기** 버튼을 눌러주세요.")


if __name__ == "__main__":
    run_app()

