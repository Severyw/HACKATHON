"""Завантаження закупівель пального з JSONL і перетворення на таблиці pandas.

Дві таблиці:
- tenders: один рядок = одна закупівля (замовник, переможець, учасники, суми);
- items:   один рядок = один товар у закупівлі (вид пального, літри, ціна за літр).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROZORRO_URL = "https://prozorro.gov.ua/tender/{}"
NO_REGION = "Область не вказана"  # військові замовники часто не публікують адресу
VAT_RATE = 0.20  # ПДВ на пальне


def read_jsonl(path: str | Path) -> list[dict]:
    """Читає файл, де кожен рядок — одна закупівля у форматі JSON."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _active_award(tender: dict) -> dict | None:
    """Рішення про переможця (award зі статусом active), якщо воно є."""
    return next((a for a in tender.get("awards", []) if a.get("status") == "active"), None)


def _winning_bid(tender: dict) -> dict | None:
    """Пропозиція переможця: саме в ній є ціна за одиницю кожного товару."""
    award = _active_award(tender) or {}
    return next((b for b in tender.get("bids", []) if b.get("id") == award.get("bid_id")), None)


def _count_bidders(tender: dict) -> int:
    """Кількість унікальних учасників (за кодом ЄДРПОУ)."""
    return len({
        tenderer["identifier"]["id"]
        for bid in tender.get("bids", [])
        for tenderer in bid.get("tenderers", [])
    })


def _period_days(period: dict | None) -> float | None:
    """Скільки днів давали на подання пропозицій."""
    if not period or not period.get("startDate") or not period.get("endDate"):
        return None
    return (pd.Timestamp(period["endDate"]) - pd.Timestamp(period["startDate"])).total_seconds() / 86400


def flatten_tender(tender: dict) -> dict:
    """Вкладений JSON закупівлі → один плаский рядок таблиці tenders."""
    buyer = tender.get("procuringEntity", {})
    award = _active_award(tender) or {}
    supplier = (award.get("suppliers") or [{}])[0]
    return {
        "id": tender["id"],
        "tender_id": tender.get("tenderID"),
        "title": tender.get("title"),
        "method": tender.get("procurementMethodType"),
        "date": pd.Timestamp(tender.get("date")),
        "buyer": buyer.get("name"),
        "buyer_id": buyer.get("identifier", {}).get("id"),
        "region": buyer.get("address", {}).get("region") or NO_REGION,
        "currency": tender.get("value", {}).get("currency"),
        "expected_value": tender.get("value", {}).get("amount"),
        "final_value": award.get("value", {}).get("amount"),
        "vat_included": bool(award.get("value", {}).get("valueAddedTaxIncluded")),
        "winner": supplier.get("name"),
        "winner_id": supplier.get("identifier", {}).get("id"),
        "bidders": _count_bidders(tender),
        "period_days": _period_days(tender.get("tenderPeriod")),
        "url": PROZORRO_URL.format(tender.get("tenderID")),
    }


def flatten_items(tender: dict) -> list[dict]:
    """Товари закупівлі з ціною за одиницю з пропозиції переможця.

    Якщо у пропозиції ціни за одиницю немає, а товар один, ділимо суму договору на кількість
    (і прибираємо ПДВ, якщо сума вказана з ним).
    """
    bid_prices = {
        item.get("id"): ((item.get("unit") or {}).get("value") or {}).get("amount")
        for item in (_winning_bid(tender) or {}).get("items", [])
    }
    items = tender.get("items", [])
    award_value = (_active_award(tender) or {}).get("value") or {}
    rows = []
    for item in items:
        price = bid_prices.get(item.get("id"))
        if price is None and len(items) == 1 and award_value.get("amount") and item.get("quantity"):
            price = award_value["amount"] / item["quantity"]
            if award_value.get("valueAddedTaxIncluded"):
                price /= 1 + VAT_RATE  # порівнюємо всі ціни без ПДВ
        rows.append({
            "id": tender["id"],
            "description": item.get("description"),
            "cpv": (item.get("classification") or {}).get("id"),
            "unit": (item.get("unit") or {}).get("name"),
            "quantity": item.get("quantity"),
            "unit_price": price,
        })
    return rows


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Головна функція модуля: файл → (tenders, items). Лише гривні та закупівлі з переможцем."""
    raw = list({t["id"]: t for t in read_jsonl(path)}.values())  # кожна закупівля один раз
    tenders = pd.DataFrame([flatten_tender(t) for t in raw])
    tenders = tenders[(tenders["currency"] == "UAH") & tenders["final_value"].notna()
                      & (tenders["expected_value"] > 0)].reset_index(drop=True)
    tenders["savings_pct"] = (tenders["expected_value"] - tenders["final_value"]) / tenders["expected_value"] * 100

    # Однакові рядки товарів не прибираємо: це можуть бути два однакові лоти в одній закупівлі
    items = pd.DataFrame([row for t in raw for row in flatten_items(t)])
    items = items[items["id"].isin(tenders["id"])].reset_index(drop=True)
    return tenders, items
