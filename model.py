from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, List[str]]:
    """
    피처가 포함된 DataFrame을 받아 랜덤포레스트 분류 모델을 학습한다.
    시간 순서를 유지하기 위해 shuffle=False로 train/test를 분리한다.
    """
    feature_cols: List[str] = [
        "return",
        "ma5",
        "ma20",
        "ma_diff",
        "vol_ma5",
        "rsi14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_width",
    ]

    X = df[feature_cols]
    y = df["target"]

    # 시간 순서 유지를 위해 셔플 없이 뒤쪽 일부를 테스트로 사용
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"테스트 정확도: {acc:.3f}")
    print("분류 리포트:")
    print(classification_report(y_test, y_pred, digits=3))

    return model, feature_cols


def predict_next_day(
    model: RandomForestClassifier,
    feature_cols: List[str],
    df: pd.DataFrame,
    ticker: str,
) -> None:
    """
    마지막 날짜 기준으로 다음 날 방향(상승/하락 또는 보합)을 예측한다.
    """
    if df.empty:
        raise ValueError("예측에 사용할 데이터가 비어 있습니다.")

    last_row = df.iloc[-1]
    X_last = last_row[feature_cols].values.reshape(1, -1)

    pred = model.predict(X_last)[0]
    proba = model.predict_proba(X_last)[0]

    # pred가 0 또는 1이므로 해당 클래스의 확률을 사용
    pred_proba = float(proba[int(pred)])
    direction = "상승" if pred == 1 else "하락 또는 보합"

    print(f"티커: {ticker}")
    print(f"예측 기준 일자: {df.index[-1].date()}")
    print(f"내일 예측 방향: {direction} (확률: {pred_proba:.2f})")

