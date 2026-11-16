"""Збирає всі кроки аналізу в одну функцію."""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from tenderguard.loader import load_data
from tenderguard.prices import add_price_stats, summarize_by_tender
from tenderguard.prozorro_api import DATA_PATH, download
from tenderguard.rules import apply_rules


class Analysis(NamedTuple):
    tenders: pd.DataFrame  # закупівлі з Risk Score і поясненнями
    items: pd.DataFrame    # товари з ціною за літр і нормою


def analyze(path: str | Path = DATA_PATH) -> Analysis:
    path = Path(path)
    if not path.exists():  # файлу немає — завантажуємо свіжі дані з API Prozorro
        download(path=path)
    tenders, items = load_data(path)
    items = add_price_stats(items, tenders)
    # Лишаємо лише товари, які можна чесно порівняти: відомий вид пального, ціна за літр і справедлива ціна.
    # Прибираємо: «бензин» без марки, нерозпізнані товари, кілька видів пального однією сумою,
    # пальне, якого замало для порівняння (А-98).
    priced = items["fuel_type"].notna() & items["unit_price"].notna() & items["fair_price"].notna()
    tenders = tenders[tenders["id"].isin(items.loc[priced, "id"])].reset_index(drop=True)
    items = items[priced].reset_index(drop=True)
    tenders = tenders.merge(summarize_by_tender(items), on="id", how="left")
    tenders["price_anomaly"] = tenders["price_anomaly"].fillna(False).astype(bool)
    tenders["overpay_uah"] = tenders["overpay_uah"].fillna(0)
    tenders = apply_rules(tenders)
    tenders = tenders.sort_values(["risk_score", "overpay_uah"], ascending=False).reset_index(drop=True)
    return Analysis(tenders, items)
