"""Завантаження завершених закупівель пального з відкритого API Prozorro у data/fuel.jsonl.

Запуск: python -m tenderguard.prozorro_api 3000
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import requests

BASE_URL = "https://public-api.prozorro.gov.ua/api/2.5"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "fuel.jsonl"
# Лише конкурентні процедури: у «звітах» про прямі договори немає учасників
COMPETITIVE = {"aboveThreshold", "belowThreshold", "priceQuotation", "aboveThresholdUA", "aboveThresholdEU"}
# Коди ДК 021 (CPV) пального: 0913… — нафтопродукти (бензин, дизель), 09122… — пропан і бутан
FUEL_CPV_PREFIXES = ("0913", "09122")


def iter_completed_ids() -> Iterator[list[str]]:
    """Гортає стрічку закупівель від найновіших; повертає сторінками ID завершених конкурентних."""
    url = f"{BASE_URL}/tenders"
    params = {"descending": 1, "limit": 1000, "opt_fields": "status,procurementMethodType"}
    while True:
        page = requests.get(url, params=params, timeout=30).json()
        if not page["data"]:
            return
        yield [
            t["id"] for t in page["data"]
            if t.get("status") == "complete" and t.get("procurementMethodType") in COMPETITIVE
        ]
        url, params = page["next_page"]["uri"], None


def fetch_tender(tender_id: str) -> dict | None:
    """Деталі однієї закупівлі. Три спроби, бо API іноді відповідає з помилкою."""
    for _ in range(3):
        try:
            response = requests.get(f"{BASE_URL}/tenders/{tender_id}", timeout=30)
            response.raise_for_status()
            return response.json()["data"]
        except (requests.RequestException, KeyError, ValueError):
            continue
    return None


def is_fuel(tender: dict) -> bool:
    """Закупівля пального: усі товари мають CPV-код пального."""
    items = tender.get("items", [])
    return bool(items) and all(
        item.get("classification", {}).get("id", "").startswith(FUEL_CPV_PREFIXES) for item in items
    )


def _organization(org: dict) -> dict:
    """Лише назва, код ЄДРПОУ та область, без контактів людей."""
    return {
        "name": org.get("name"),
        "identifier": {"id": org.get("identifier", {}).get("id")},
        "address": {"region": org.get("address", {}).get("region")},
    }


def compact(tender: dict) -> dict:
    """Залишає лише поля, потрібні для аналізу (повний запис займає ~90 КБ)."""
    keep = ("id", "tenderID", "title", "status", "procurementMethodType",
            "mainProcurementCategory", "value", "tenderPeriod", "dateCreated", "date")
    return {
        **{key: tender.get(key) for key in keep},
        "procuringEntity": _organization(tender.get("procuringEntity", {})),
        "items": [
            {k: item.get(k) for k in ("id", "description", "quantity", "unit", "classification")}
            for item in tender.get("items", [])
        ],
        "bids": [
            {"id": b.get("id"), "status": b.get("status"), "value": b.get("value"),
             "tenderers": [_organization(t) for t in b.get("tenderers", [])],
             # ціна за одиницю кожного товару в пропозиції
             "items": [{"id": i.get("id"), "quantity": i.get("quantity"), "unit": i.get("unit")}
                       for i in b.get("items", [])]}
            for b in tender.get("bids", [])
        ],
        "awards": [
            {"status": a.get("status"), "value": a.get("value"), "bid_id": a.get("bid_id"),
             "suppliers": [_organization(s) for s in a.get("suppliers", [])]}
            for a in tender.get("awards", [])
        ],
    }


def download(limit: int = 3000, path: Path = DATA_PATH, workers: int = 24) -> int:
    """Качає закупівлі пального, поки не набере limit. Деталі тягне паралельно."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = scanned = 0
    with open(path, "w", encoding="utf-8") as file, ThreadPoolExecutor(max_workers=workers) as pool:
        for ids in iter_completed_ids():
            for tender in pool.map(fetch_tender, ids):
                scanned += 1
                if tender and is_fuel(tender):
                    file.write(json.dumps(compact(tender), ensure_ascii=False) + "\n")
                    saved += 1
            print(f"переглянуто {scanned}, пального {saved}", flush=True)
            if saved >= limit:
                break
    return saved


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    print(f"Збережено {download(count)} закупівель пального у {DATA_PATH}")

