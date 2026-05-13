from __future__ import annotations


import json

import logging

import re

import sqlite3

from collections import Counter

from datetime import datetime

from pathlib import Path

from typing import Any, Dict, Iterable, List, Tuple


import numpy as np

import pandas as pd


from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.cluster import DBSCAN, KMeans

from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import (

    accuracy_score,

    classification_report,

    f1_score,

    precision_score,

    recall_score,

    roc_auc_score,

    roc_curve,

)

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline


try:

    from xgboost import XGBClassifier

except Exception:

    XGBClassifier = None


try:

    from lightgbm import LGBMClassifier

except Exception:

    LGBMClassifier = None


ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "reviews_dataset.csv"

SQLITE_PATH = ROOT / "data" / "reviews.sqlite"

EXPORT_DIR = ROOT / "exports"

EXPORT_DIR.mkdir(exist_ok=True)

LOG_PATH = ROOT / "app.log"


class DenseTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self


    def transform(self, X):

        return X.toarray() if hasattr(X, "toarray") else X


logger = logging.getLogger("ml_product")

logger.setLevel(logging.INFO)

logger.propagate = False

if not logger.handlers:

    _handler = logging.FileHandler(LOG_PATH, encoding="utf-8")

    _handler.setFormatter(logging.Formatter("%(asctime)s | user_id=%(user_id)s | action=%(action)s | result=%(result)s"))

    logger.addHandler(_handler)


RU_STOPWORDS = {

    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она",

    "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее",

    "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда",

    "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него",

    "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего",

    "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их",

    "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под",

    "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем",

    "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас",

    "были", "куда", "зачем", "сказать", "всех", "никогда", "сегодня", "можно", "при", "наконец",

    "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти", "нас",

    "про", "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою",

    "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более",

    "всегда", "конечно", "всю", "между", "товар", "покупка", "пользуюсь", "особенно", "заметно",

}


def log_action(user_id: str, action: str, result: str) -> None:

    logger.info("event", extra={"user_id": user_id, "action": action, "result": result})


def mask_phone(text: str) -> str:

    phone_pattern = re.compile(r"(?:\+7|8)[\s\-\(]*\d{3}[\s\-\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}")

    return phone_pattern.sub("[PHONE]", text)


def mask_name(text: str) -> str:

    name_pattern = re.compile(r"(меня зовут|я\s+—|я\s+-|я\s+)(\s*)([А-ЯЁ][а-яё]{2,})", flags=re.IGNORECASE)

    return name_pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[NAME]", text)


def anonymize_text(text: str) -> str:

    return mask_name(mask_phone(str(text)))


def load_reviews(path: Path | str = DATA_PATH) -> pd.DataFrame:

    df = pd.read_csv(path)

    required = {"review_id", "user_id", "product_id", "product", "text", "rating", "date"}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df.copy()

    df["text"] = df["text"].map(anonymize_text)

    df["rating"] = df["rating"].astype(int)

    df["date"] = pd.to_datetime(df["date"])

    df["sentiment"] = np.where(df["rating"] >= 4, 1, 0)

    df["sentiment_label"] = np.where(df["sentiment"] == 1, "позитивный", "негативный")

    return df


def init_sqlite(csv_path: Path | str = DATA_PATH, sqlite_path: Path | str = SQLITE_PATH) -> None:

    df = load_reviews(csv_path)

    con = sqlite3.connect(sqlite_path)

    df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_sql("reviews", con, if_exists="replace", index=False)

    con.close()


def add_review(user_id: str, product_id: str, product: str, text: str, rating: int, sqlite_path: Path | str = SQLITE_PATH) -> None:

    text = anonymize_text(text)

    con = sqlite3.connect(sqlite_path)

    row = {

        "review_id": f"R{datetime.now().strftime('%Y%m%d%H%M%S%f')}",

        "user_id": user_id,

        "product_id": product_id,

        "product": product,

        "text": text,

        "rating": int(rating),

        "date": datetime.now().strftime("%Y-%m-%d"),

        "sentiment": 1 if int(rating) >= 4 else 0,

        "sentiment_label": "позитивный" if int(rating) >= 4 else "негативный",

    }

    pd.DataFrame([row]).to_sql("reviews", con, if_exists="append", index=False)

    con.close()

    log_action(user_id, "add_review", f"rating={rating}; product={product_id}")


def tokenize_ru(text: str) -> List[str]:

    words = re.findall(r"[а-яёa-z]{3,}", str(text).lower())

    return [w for w in words if w not in RU_STOPWORDS]


def top_negative_words(df: pd.DataFrame, n: int = 5) -> List[Tuple[str, int]]:

    negative = df[df["sentiment"] == 0]["text"].tolist()

    counts: Counter[str] = Counter()

    for text in negative:

        counts.update(tokenize_ru(text))

    return counts.most_common(n)


def top_positive_products(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:

    stats = (

        df.groupby(["product_id", "product"])

        .agg(reviews_count=("review_id", "count"), positive_share=("sentiment", "mean"), avg_rating=("rating", "mean"))

        .reset_index()

    )

    stats["positive_share"] = (stats["positive_share"] * 100).round(1)

    stats["avg_rating"] = stats["avg_rating"].round(2)

    return stats.sort_values(["positive_share", "reviews_count"], ascending=False).head(n)


def product_statistics(df: pd.DataFrame) -> pd.DataFrame:

    stats = (

        df.groupby(["product_id", "product"])

        .agg(

            reviews_count=("review_id", "count"),

            avg_rating=("rating", "mean"),

            positive_share=("sentiment", "mean"),

            last_review=("date", "max"),

        )

        .reset_index()

    )

    stats["avg_rating"] = stats["avg_rating"].round(2)

    stats["positive_share"] = (stats["positive_share"] * 100).round(1)

    return stats.sort_values("avg_rating", ascending=False)


def get_model_specs() -> Dict[str, Tuple[Any, Dict[str, list]]]:

    specs: Dict[str, Tuple[Any, Dict[str, list]]] = {

        "LogisticRegression": (

            LogisticRegression(max_iter=500, class_weight="balanced"),

            {"clf__C": [0.5, 1.0], "clf__solver": ["liblinear"], "clf__class_weight": ["balanced"]},

        ),

        "RandomForest": (

            RandomForestClassifier(random_state=42, class_weight="balanced"),

            {"clf__n_estimators": [60], "clf__max_depth": [None, 8], "clf__min_samples_split": [2, 4]},

        ),

        "MLPClassifier": (

            MLPClassifier(max_iter=80, random_state=42, early_stopping=True, n_iter_no_change=5),

            {"clf__hidden_layer_sizes": [(16,), (24,)], "clf__alpha": [0.001], "clf__learning_rate_init": [0.001, 0.01]},

        ),

    }

    if XGBClassifier is not None:

        specs["XGBoost"] = (

            XGBClassifier(

                random_state=42,

                eval_metric="logloss",

                n_jobs=1,

                verbosity=0,

            ),

            {"clf__n_estimators": [40], "clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1]},

        )

    if LGBMClassifier is not None:

        specs["LightGBM"] = (

            LGBMClassifier(random_state=42, n_jobs=1, verbose=-1),

            {"clf__n_estimators": [40], "clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1]},

        )

    return specs


def _evaluate_model(model: Any, x_test: Iterable[str], y_test: Iterable[int]) -> Dict[str, float]:

    pred = model.predict(x_test)

    result = {

        "Accuracy": accuracy_score(y_test, pred),

        "Precision": precision_score(y_test, pred, zero_division=0),

        "Recall": recall_score(y_test, pred, zero_division=0),

        "F1": f1_score(y_test, pred, zero_division=0),

    }

    try:

        proba = model.predict_proba(x_test)[:, 1]

        result["ROC_AUC"] = roc_auc_score(y_test, proba)

    except Exception:

        result["ROC_AUC"] = np.nan

    return {k: round(float(v), 4) for k, v in result.items()}


def train_and_compare_models(df: pd.DataFrame, with_grid: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    x_train, x_test, y_train, y_test = train_test_split(

        df["text"], df["sentiment"], test_size=0.25, random_state=42, stratify=df["sentiment"]

    )

    rows: List[Dict[str, Any]] = []

    trained: Dict[str, Any] = {}

    for name, (clf, params) in get_model_specs().items():

        base_pipe = Pipeline([

            ("tfidf", TfidfVectorizer(max_features=350, ngram_range=(1, 2), min_df=2)),

            ("dense", DenseTransformer()),

            ("clf", clf),

        ])

        base_pipe.fit(x_train, y_train)

        row = {"model": name, "stage": "до подбора", **_evaluate_model(base_pipe, x_test, y_test)}

        rows.append(row)


        if with_grid:

            grid = GridSearchCV(base_pipe, params, scoring="f1", cv=3, n_jobs=1)

            grid.fit(x_train, y_train)

            best = grid.best_estimator_

            tuned_row = {"model": name, "stage": "после GridSearchCV", **_evaluate_model(best, x_test, y_test), "best_params": json.dumps(grid.best_params_, ensure_ascii=False)}

            rows.append(tuned_row)

            trained[name] = best

        else:

            trained[name] = base_pipe

    metrics = pd.DataFrame(rows)

    metrics.to_csv(EXPORT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    return metrics, trained


def get_best_model(metrics: pd.DataFrame, trained_models: Dict[str, Any]) -> Any:

    after = metrics[metrics["stage"].str.contains("после")]

    if len(after) == 0:

        after = metrics

    best_name = after.sort_values("F1", ascending=False).iloc[0]["model"]

    return trained_models[best_name]


def predict_sentiment(text: str, model: Any | None = None, df: pd.DataFrame | None = None) -> Dict[str, Any]:

    if df is None:

        df = load_reviews()

    if model is None:

        metrics, models = train_and_compare_models(df, with_grid=False)

        model = get_best_model(metrics, models)

    safe_text = anonymize_text(text)

    label_num = int(model.predict([safe_text])[0])

    if hasattr(model, "predict_proba"):

        confidence = float(model.predict_proba([safe_text])[0][label_num]) * 100

    else:

        confidence = 50.0

    label = "позитивный" if label_num == 1 else "негативный"

    return {"label": label, "confidence": round(confidence, 1), "masked_text": safe_text}


def recommend_for_user(df: pd.DataFrame, user_id: str, n: int = 5) -> pd.DataFrame:

    matrix = df.pivot_table(index="user_id", columns="product_id", values="rating", aggfunc="mean")

    if user_id not in matrix.index:

        return product_statistics(df).head(n)[["product_id", "product", "avg_rating", "positive_share"]]

    user_vec = matrix.loc[user_id]

    centered = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0)

    target = centered.loc[user_id].values

    sims = {}

    for other in centered.index:

        if other == user_id:

            continue

        ov = centered.loc[other].values

        denom = np.linalg.norm(target) * np.linalg.norm(ov)

        sims[other] = 0.0 if denom == 0 else float(np.dot(target, ov) / denom)

    similar_users = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:7]

    rated = set(user_vec.dropna().index)

    candidates = []

    product_names = df.drop_duplicates("product_id").set_index("product_id")["product"].to_dict()

    global_avg = df.groupby("product_id")["rating"].mean()

    for product_id in matrix.columns:

        if product_id in rated:

            continue

        numerator = 0.0

        denominator = 0.0

        for other, sim in similar_users:

            rating = matrix.loc[other, product_id]

            if pd.notna(rating) and sim > 0:

                numerator += sim * rating

                denominator += abs(sim)

        score = numerator / denominator if denominator else float(global_avg.get(product_id, 3.0))

        candidates.append({"product_id": product_id, "product": product_names.get(product_id, product_id), "predicted_rating": round(score, 2)})

    if not candidates:

        return product_statistics(df).head(n)[["product_id", "product", "avg_rating", "positive_share"]]

    return pd.DataFrame(candidates).sort_values("predicted_rating", ascending=False).head(n)


def forecast_review_counts(df: pd.DataFrame, days_forward: int = 7) -> pd.DataFrame:

    daily = df.groupby(df["date"].dt.date).size().rename("reviews_count").reset_index()

    daily["day_num"] = np.arange(len(daily))

    model = LinearRegression().fit(daily[["day_num"]], daily["reviews_count"])

    future_rows = []

    last_date = pd.to_datetime(daily["date"].max())

    for i in range(1, days_forward + 1):

        day_num = len(daily) + i - 1

        pred = max(0, float(model.predict([[day_num]])[0]))

        future_rows.append({"date": (last_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d"), "forecast_reviews": round(pred, 1)})

    result = pd.DataFrame(future_rows)

    result.to_csv(EXPORT_DIR / "review_forecast.csv", index=False, encoding="utf-8-sig")

    return result


def vectorize_reviews_sentence_transformer(texts: List[str]) -> Tuple[np.ndarray, str]:

    try:

        from sentence_transformers import SentenceTransformer


        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        return model.encode(texts, show_progress_bar=False), "SentenceTransformer: paraphrase-multilingual-MiniLM-L12-v2"

    except Exception:

        tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2)

        x = tfidf.fit_transform(texts)

        n_components = min(50, max(2, x.shape[1] - 1))

        svd = TruncatedSVD(n_components=n_components, random_state=42)

        return svd.fit_transform(x), "Fallback: TF-IDF + TruncatedSVD (для офлайн-запуска)"


def cluster_reviews(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:

    embeddings, vectorizer_name = vectorize_reviews_sentence_transformer(df["text"].tolist())

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

    dbscan = DBSCAN(eps=0.8, min_samples=5)

    labels_km = kmeans.fit_predict(embeddings)

    labels_db = dbscan.fit_predict(embeddings)

    pca = PCA(n_components=2, random_state=42)

    coords = pca.fit_transform(embeddings)


    clustered = df.copy()

    clustered["kmeans_cluster"] = labels_km

    clustered["dbscan_cluster"] = labels_db

    clustered["x"] = coords[:, 0]

    clustered["y"] = coords[:, 1]


    rows = []

    for cluster_id, group in clustered.groupby("kmeans_cluster"):

        counts: Counter[str] = Counter()

        for text in group["text"]:

            counts.update(tokenize_ru(text))

        rows.append({

            "cluster": int(cluster_id),

            "reviews_count": int(len(group)),

            "positive_share": round(float(group["sentiment"].mean() * 100), 1),

            "avg_rating": round(float(group["rating"].mean()), 2),

            "top_10_words": ", ".join([w for w, _ in counts.most_common(10)]),

        })

    summary = pd.DataFrame(rows).sort_values("cluster")

    clustered.to_csv(EXPORT_DIR / "clustered_reviews.csv", index=False, encoding="utf-8-sig")

    summary.to_csv(EXPORT_DIR / "cluster_summary.csv", index=False, encoding="utf-8-sig")

    return clustered, summary, vectorizer_name


if __name__ == "__main__":

    df_main = load_reviews()

    metrics_main, models_main = train_and_compare_models(df_main, with_grid=True)

    print(metrics_main)

    print("Best prediction:", predict_sentiment("Отличный товар, доставка быстрая, всё понравилось", get_best_model(metrics_main, models_main), df_main))
