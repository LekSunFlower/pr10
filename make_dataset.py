from __future__ import annotations


import random

import sqlite3

from datetime import datetime, timedelta

from pathlib import Path


import pandas as pd


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"

DATA_DIR.mkdir(exist_ok=True)


PRODUCTS = [

    ("P001", "Смартфон Aurora X1"),

    ("P002", "Ноутбук VectorBook 14"),

    ("P003", "Беспроводные наушники SonicPods"),

    ("P004", "Умные часы FitTime Pro"),

    ("P005", "Пылесос CleanBot Mini"),

    ("P006", "Кофемашина AromaBarista"),

    ("P007", "Электрочайник GlassHeat"),

    ("P008", "Игровая мышь CyberClick"),

    ("P009", "Клавиатура KeyMaster RGB"),

    ("P010", "Монитор VisionView 27"),

    ("P011", "Планшет TabLine 10"),

    ("P012", "Роутер HomeNet AX"),

]


POSITIVE = [

    "Отличный товар, работает быстро и стабильно, покупкой довольна.",

    "Качество выше ожиданий, упаковка аккуратная, доставка пришла вовремя.",

    "Очень удобная вещь, пользуюсь каждый день, рекомендую к покупке.",

    "Хорошая сборка, приятный дизайн, цена полностью оправдана.",

    "Товар понравился, всё подключилось без проблем, инструкция понятная.",

    "За свои деньги просто супер, функциональность отличная, нареканий нет.",

    "Покупка удачная, устройство тихое, красивое и надежное.",

    "Сервис магазина порадовал, товар новый, без дефектов и царапин.",

    "Работает лучше предыдущей модели, батарея держит долго, впечатления положительные.",

    "Очень довольна заказом, товар соответствует описанию и фото.",

]


NEGATIVE = [

    "Товар разочаровал, качество слабое, появились проблемы уже в первый день.",

    "Доставка задержалась, коробка была мятая, впечатление испорчено.",

    "Не рекомендую, устройство шумит, иногда зависает и плохо работает.",

    "Цена завышена, ожидала большего, функциональность неудобная.",

    "Пришёл брак, пришлось писать в поддержку и оформлять возврат.",

    "Пластик дешевый, сборка хлипкая, товар не стоит своих денег.",

    "Описание на сайте красивое, но в реальности товар заметно хуже.",

    "Покупкой недовольна, заряд держит мало, подключение часто пропадает.",

    "Плохой опыт, после недели использования появилась неисправность.",

    "Не подошло, работает нестабильно, интерфейс неудобный и медленный.",

]


NEUTRAL = [

    "Обычный товар, в целом работает, но без сильных преимуществ.",

    "Нормально за свою цену, есть плюсы и минусы.",

    "Покупка средняя, ожидания частично оправдались.",

    "Можно пользоваться, но качество не идеальное.",

]


ADD_POS = ["быстро", "удобно", "надежно", "приятно", "выгодно", "качественно", "аккуратно", "стильно"]

ADD_NEG = ["брак", "шум", "задержка", "царапина", "возврат", "поломка", "медленно", "неудобно"]

NAMES = ["Анна", "Мария", "Елена", "Ольга", "Дарья", "Ирина", "Сергей", "Алексей", "Иван", "Никита"]


random.seed(42)

rows = []

start = datetime(2026, 1, 1)

users = [f"U{str(i).zfill(3)}" for i in range(1, 46)]


for i in range(1, 321):

    product_id, product = random.choice(PRODUCTS)

    user_id = random.choice(users)

    date = start + timedelta(days=random.randint(0, 115))


    if product_id in {"P001", "P003", "P010"}:

        rating = random.choices([5, 4, 3, 2, 1], weights=[42, 30, 14, 8, 6], k=1)[0]

    elif product_id in {"P005", "P007"}:

        rating = random.choices([5, 4, 3, 2, 1], weights=[18, 20, 24, 20, 18], k=1)[0]

    else:

        rating = random.choices([5, 4, 3, 2, 1], weights=[28, 26, 20, 16, 10], k=1)[0]


    if rating >= 4:

        base = random.choice(POSITIVE)

        tail = ", ".join(random.sample(ADD_POS, 2))

    elif rating <= 2:

        base = random.choice(NEGATIVE)

        tail = ", ".join(random.sample(ADD_NEG, 2))

    else:

        base = random.choice(NEUTRAL)

        tail = random.choice(ADD_POS) + ", но " + random.choice(ADD_NEG)


    text = f"{base} Особенно заметно: {tail}."


    if i in {7, 41, 89, 144, 205}:

        text += f" Меня зовут {random.choice(NAMES)}, телефон +7 999 {random.randint(100,999)}-{random.randint(10,99)}-{random.randint(10,99)}."

    if i in {62, 170, 250}:

        text += f" Свяжитесь со мной по номеру 8-916-{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(10,99)}."


    rows.append({

        "review_id": f"R{str(i).zfill(4)}",

        "user_id": user_id,

        "product_id": product_id,

        "product": product,

        "text": text,

        "rating": rating,

        "date": date.strftime("%Y-%m-%d"),

    })


df = pd.DataFrame(rows)

df.to_csv(DATA_DIR / "reviews_dataset.csv", index=False, encoding="utf-8-sig")


conn = sqlite3.connect(DATA_DIR / "reviews.sqlite")

df.to_sql("reviews", conn, if_exists="replace", index=False)

conn.close()

print(f"Saved {len(df)} rows to {DATA_DIR / 'reviews_dataset.csv'} and SQLite")
