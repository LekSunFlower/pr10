from __future__ import annotations


import sqlite3

from pathlib import Path


import matplotlib.pyplot as plt

import pandas as pd

from wordcloud import WordCloud


from ml_product import (

    DATA_PATH,

    SQLITE_PATH,

    EXPORT_DIR,

    cluster_reviews,

    forecast_review_counts,

    get_best_model,

    init_sqlite,

    load_reviews,

    predict_sentiment,

    product_statistics,

    recommend_for_user,

    top_negative_words,

    top_positive_products,

    train_and_compare_models,

)


ROOT = Path(__file__).resolve().parent

IMG_DIR = ROOT / "static" / "img"

IMG_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_DIR.mkdir(exist_ok=True)

FONT_PATH = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"


def save_rating_distribution(df: pd.DataFrame) -> None:

    counts = df["rating"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))

    plt.bar(counts.index.astype(str), counts.values)

    plt.title("Распределение оценок клиентов")

    plt.xlabel("Оценка")

    plt.ylabel("Количество отзывов")

    plt.tight_layout()

    plt.savefig(IMG_DIR / "ratings_distribution.png", dpi=160)

    plt.close()


def save_reviews_by_day(df: pd.DataFrame) -> None:

    daily = df.groupby(df["date"].dt.date).size()

    rolling = daily.rolling(7, min_periods=1).mean()

    plt.figure(figsize=(9, 4.8))

    plt.plot(pd.to_datetime(daily.index), daily.values, label="Отзывы за день")

    plt.plot(pd.to_datetime(rolling.index), rolling.values, label="Скользящее среднее 7 дней")

    plt.title("Динамика количества отзывов по дням")

    plt.xlabel("Дата")

    plt.ylabel("Количество")

    plt.legend()

    plt.xticks(rotation=35, ha="right")

    plt.tight_layout()

    plt.savefig(IMG_DIR / "reviews_by_day.png", dpi=160)

    plt.close()


def save_wordcloud(df: pd.DataFrame) -> None:

    text = " ".join(df["text"].astype(str).tolist())

    wc = WordCloud(width=1000, height=550, background_color="white", font_path=FONT_PATH, collocations=False).generate(text)

    plt.figure(figsize=(10, 5.5))

    plt.imshow(wc, interpolation="bilinear")

    plt.axis("off")

    plt.title("Облако слов по отзывам")

    plt.tight_layout()

    plt.savefig(IMG_DIR / "wordcloud.png", dpi=160)

    plt.close()


def save_top_products(df: pd.DataFrame) -> None:

    top = top_positive_products(df, 8).sort_values("positive_share")

    plt.figure(figsize=(8.5, 5))

    plt.barh(top["product"], top["positive_share"])

    plt.title("Топ товаров по доле позитивных отзывов")

    plt.xlabel("Доля позитива, %")

    plt.tight_layout()

    plt.savefig(IMG_DIR / "top_positive_products.png", dpi=160)

    plt.close()


def save_model_metrics(metrics: pd.DataFrame) -> None:

    after = metrics[metrics["stage"].str.contains("после")].copy()

    after = after[["model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]

    fig, ax = plt.subplots(figsize=(10, 3.5))

    ax.axis("off")

    table = ax.table(cellText=after.round(4).values, colLabels=after.columns, loc="center")

    table.auto_set_font_size(False)

    table.set_fontsize(9)

    table.scale(1, 1.4)

    plt.title("Сравнение моделей после GridSearchCV")

    plt.tight_layout()

    plt.savefig(IMG_DIR / "model_metrics_table.png", dpi=170)

    plt.close()


def save_roc_curves(df: pd.DataFrame, models: dict) -> None:

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import roc_curve


    _, x_test, _, y_test = train_test_split(df["text"], df["sentiment"], test_size=0.25, random_state=42, stratify=df["sentiment"])

    plt.figure(figsize=(7, 5.5))

    for name, model in models.items():

        if not hasattr(model, "predict_proba"):

            continue

        try:

            proba = model.predict_proba(x_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, proba)

            plt.plot(fpr, tpr, label=name)

        except Exception:

            continue

    plt.plot([0, 1], [0, 1], linestyle="--", label="Случайная модель")

    plt.title("ROC-кривые моделей")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend(fontsize=8)

    plt.tight_layout()

    plt.savefig(IMG_DIR / "roc_curves.png", dpi=160)

    plt.close()


def save_clusters(clustered: pd.DataFrame) -> None:

    plt.figure(figsize=(7.5, 5.5))

    scatter = plt.scatter(clustered["x"], clustered["y"], c=clustered["kmeans_cluster"], s=36, alpha=0.82)

    plt.title("Кластеры отзывов после векторизации")

    plt.xlabel("PCA 1")

    plt.ylabel("PCA 2")

    plt.colorbar(scatter, label="KMeans cluster")

    plt.tight_layout()

    plt.savefig(IMG_DIR / "clusters.png", dpi=160)

    plt.close()


    try:

        import plotly.express as px

        fig = px.scatter(

            clustered,

            x="x",

            y="y",

            color=clustered["kmeans_cluster"].astype(str),

            hover_data=["product", "rating", "sentiment_label", "text"],

            title="Интерактивная визуализация кластеров отзывов",

        )

        fig.write_html(EXPORT_DIR / "interactive_clusters.html", include_plotlyjs="cdn")

    except Exception:

        pass


def create_student_report_png(df: pd.DataFrame, metrics: pd.DataFrame) -> None:

    top = top_positive_products(df, 5)

    neg = top_negative_words(df, 5)

    forecast = forecast_review_counts(df, 7)

    fig = plt.figure(figsize=(14, 10))


    ax1 = fig.add_subplot(2, 2, 1)

    counts = df["rating"].value_counts().sort_index()

    ax1.bar(counts.index.astype(str), counts.values)

    ax1.set_title("Распределение оценок")

    ax1.set_xlabel("Оценка")

    ax1.set_ylabel("Количество")


    ax2 = fig.add_subplot(2, 2, 2)

    top_sorted = top.sort_values("positive_share")

    ax2.barh(top_sorted["product"], top_sorted["positive_share"])

    ax2.set_title("Топ товаров по доле позитива")

    ax2.set_xlabel("Доля позитива, %")


    ax3 = fig.add_subplot(2, 2, 3)

    ax3.bar([w for w, _ in neg], [c for _, c in neg])

    ax3.set_title("5 частых слов в негативных отзывах")

    ax3.tick_params(axis="x", rotation=25)


    ax4 = fig.add_subplot(2, 2, 4)

    m = metrics[metrics["stage"].str.contains("после")][["model", "F1"]]

    ax4.barh(m["model"], m["F1"])

    ax4.set_xlim(0, 1.05)

    ax4.set_title("F1 моделей после подбора")


    fig.suptitle("student_report.png — аналитика отзывов интернет-магазина", fontsize=16)

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    plt.savefig(EXPORT_DIR / "student_report.png", dpi=160)

    plt.close()


def main() -> None:

    init_sqlite(DATA_PATH, SQLITE_PATH)

    df = load_reviews()

    metrics, models = train_and_compare_models(df, with_grid=True)

    best_model = get_best_model(metrics, models)

    pred = predict_sentiment("Отличный магазин, товар пришёл быстро и работает без проблем", best_model, df)

    pd.DataFrame([pred]).to_csv(EXPORT_DIR / "prediction_example.csv", index=False, encoding="utf-8-sig")


    save_rating_distribution(df)

    save_reviews_by_day(df)

    save_wordcloud(df)

    save_top_products(df)

    save_model_metrics(metrics)

    save_roc_curves(df, models)

    clustered, summary, vectorizer_name = cluster_reviews(df)

    save_clusters(clustered)

    create_student_report_png(df, metrics)

    product_statistics(df).to_csv(EXPORT_DIR / "product_statistics.csv", index=False, encoding="utf-8-sig")

    top_positive_products(df, 10).to_csv(EXPORT_DIR / "top_positive_products.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(top_negative_words(df, 10), columns=["word", "count"]).to_csv(EXPORT_DIR / "negative_words.csv", index=False, encoding="utf-8-sig")

    print("Графики и таблицы обновлены")


if __name__ == "__main__":

    main()
