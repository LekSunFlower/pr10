from __future__ import annotations


import uuid

from pathlib import Path


import pandas as pd

from flask import Flask, redirect, render_template, request, session, url_for


from ml_product import (

    LOG_PATH,

    SQLITE_PATH,

    add_review,

    get_best_model,

    load_reviews,

    log_action,

    predict_sentiment,

    product_statistics,

    recommend_for_user,

    top_negative_words,

    train_and_compare_models,

)


app = Flask(__name__)

app.secret_key = "change-this-key-for-production"


CACHE = {"metrics": None, "models": None, "best_model": None}


def current_user_id() -> str:

    if "user_id" not in session:

        session["user_id"] = "web_" + uuid.uuid4().hex[:10]

    return session["user_id"]


def get_data() -> pd.DataFrame:

    return load_reviews()


def get_model():

    df = get_data()

    if CACHE["best_model"] is None:

        metrics, models = train_and_compare_models(df, with_grid=True)

        CACHE["metrics"] = metrics

        CACHE["models"] = models

        CACHE["best_model"] = get_best_model(metrics, models)

    return CACHE["best_model"]


@app.route("/", methods=["GET", "POST"])

def index():

    user_id = current_user_id()

    df = get_data()

    products = df[["product_id", "product"]].drop_duplicates().sort_values("product").to_dict("records")

    prediction = None

    if request.method == "POST":

        product_id = request.form.get("product_id")

        product = next((p["product"] for p in products if p["product_id"] == product_id), "Неизвестный товар")

        text = request.form.get("text", "")

        rating = int(request.form.get("rating", 3))

        prediction = predict_sentiment(text, get_model(), df)

        add_review(user_id, product_id, product, text, rating)

        log_action(user_id, "submit_review", f"{product_id}; {prediction['label']}; {prediction['confidence']}%")

        df = get_data()

    recommendations = recommend_for_user(df, user_id, 5).to_dict("records")

    neg_words = top_negative_words(df, 5)

    return render_template(

        "index.html",

        user_id=user_id,

        products=products,

        prediction=prediction,

        recommendations=recommendations,

        neg_words=neg_words,

    )


@app.route("/stats")

def stats():

    user_id = current_user_id()

    df = get_data()

    stats_df = product_statistics(df)

    log_action(user_id, "open_stats", f"{len(stats_df)} products")

    return render_template("stats.html", stats=stats_df.to_dict("records"))


@app.route("/models")

def models():

    user_id = current_user_id()

    if CACHE["metrics"] is None:

        df = get_data()

        metrics, models_dict = train_and_compare_models(df, with_grid=True)

        CACHE["metrics"] = metrics

        CACHE["models"] = models_dict

        CACHE["best_model"] = get_best_model(metrics, models_dict)

    log_action(user_id, "open_models", "metrics table")

    return render_template("models.html", metrics=CACHE["metrics"].to_dict("records"))


@app.route("/clusters")

def clusters():

    user_id = current_user_id()

    log_action(user_id, "open_clusters", "interactive plot")

    html_path = Path("exports/interactive_clusters.html")

    return render_template("clusters.html", html_path=html_path)


@app.route("/admin/logs")

def logs():

    user_id = current_user_id()

    lines = []

    if LOG_PATH.exists():

        lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[-50:]

    log_action(user_id, "open_logs", "last 50 lines")

    return render_template("logs.html", lines=lines)


if __name__ == "__main__":

    app.run()
