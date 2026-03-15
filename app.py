import streamlit as st
import pandas as pd

from data_loader import load_data
from features import add_features
from model import train_model, predict_next_day


def run_app() -> None:
    st.set_page_config(page_title="한국 주식 예측 데모", layout="wide")

    st.title("📈 한국 주식 방향 예측 데모")

    st.sidebar.header("설정")

    default_ticker = "005930.KS"
    ticker = st.sidebar.text_input("티커 (야후 형식)", value=default_ticker)
    period = st.sidebar.selectbox("기간", ["1y", "3y", "5y"], index=2)

    if st.sidebar.button("예측하기"):
        with st.spinner("데이터 다운로드 및 모델 학습 중..."):
            try:
                df_raw: pd.DataFrame = load_data(ticker=ticker, period=period)
            except Exception as e:
                st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")
                return

            df_feat: pd.DataFrame = add_features(df_raw)

            model, feature_cols = train_model(df_feat)

        st.success("모델 학습 완료!")

        st.subheader("내일 방향 예측")
        try:
            st.write(f"티커: **{ticker}**")
            predict_next_day(model, feature_cols, df_feat, ticker=ticker)
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")

        st.subheader("가격 및 이동평균")
        price_cols = ["Close", "ma5", "ma20"]
        available_cols = [c for c in price_cols if c in df_feat.columns]
        if available_cols:
            st.line_chart(df_feat[available_cols])

        if "rsi14" in df_feat.columns:
            st.subheader("RSI(14)")
            st.line_chart(df_feat[["rsi14"]])

    else:
        st.info("왼쪽 사이드바에서 티커와 기간을 선택한 뒤 **예측하기** 버튼을 눌러주세요.")


if __name__ == "__main__":
    run_app()

