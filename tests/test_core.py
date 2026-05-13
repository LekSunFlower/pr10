from __future__ import annotations


from pathlib import Path

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd


from ml_product import (

    DATA_PATH,

    load_reviews,

    mask_name,

    mask_phone,

    predict_sentiment,

    product_statistics,

    recommend_for_user,

    top_negative_words,

    train_and_compare_models,

)


def test_mask_phone():

    text = "Мой телефон +7 999 123-45-67, перезвоните."

    assert "[PHONE]" in mask_phone(text)

    assert "+7 999" not in mask_phone(text)


def test_mask_name():

    text = "Меня зовут Анна, заказ пришёл быстро."

    masked = mask_name(text)

    assert "[NAME]" in masked

    assert "Анна" not in masked


def test_load_reviews_has_required_columns():

    df = load_reviews(DATA_PATH)

    assert len(df) >= 200

    assert {"text", "rating", "product", "date", "user_id", "sentiment"}.issubset(df.columns)


def test_metrics_table_contains_f1():

    df = load_reviews(DATA_PATH).head(120)

    metrics, _ = train_and_compare_models(df, with_grid=False)

    assert "F1" in metrics.columns

    assert metrics["F1"].between(0, 1).all()


def test_predict_sentiment_returns_label_and_confidence():

    df = load_reviews(DATA_PATH).head(160)

    metrics, models = train_and_compare_models(df, with_grid=False)

    model = next(iter(models.values()))

    result = predict_sentiment("Отличный товар, работает стабильно", model=model, df=df)

    assert result["label"] in {"позитивный", "негативный"}

    assert 0 <= result["confidence"] <= 100


def test_recommendations_for_user():

    df = load_reviews(DATA_PATH)

    recs = recommend_for_user(df, df["user_id"].iloc[0], n=3)

    assert len(recs) <= 3

    assert "product" in recs.columns
