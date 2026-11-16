"""Аналіз цін на пальне: вид пального, справедлива ціна, межа аномалії та переплата.

Порівнюємо лише однакове: дизель з дизелем, А-95 з А-95, ціна за літр без ПДВ.
"""
from __future__ import annotations

import re

import pandas as pd

from tenderguard.loader import NO_REGION, VAT_RATE

MIN_ANALOGS = 10         # менше аналогів — порівняння ненадійне
MIN_REGION_ANALOGS = 5   # менше аналогів в області — порівняння з областю не робимо
NORMAL_SPREAD_PCT = 5    # до +5% над справедливою ціною — звичайний розкид, переплатою не вважаємо
COMPETITIVE_BIDDERS = 3  # «реальна конкуренція» — від 3 учасників
VOLUME_BINS = [0, 2_000, 10_000, float("inf")]
VOLUME_LABELS = ["до 2 тис. л", "2–10 тис. л", "понад 10 тис. л"]

# Точні коди CPV, якщо замовник їх вказав (це трапляється лише у ~4% товарів)
CPV_FUEL = {"09134": "Дизель", "09133": "Газ", "09122": "Газ"}
NOT_COMPARED_CPV = ("09131",)  # авіаційне паливо: окремий ринок, не порівнюємо

# Порядок важливий: спершу конкретні марки бензину, потім загальні слова
FUEL_PATTERNS = {
    "А-98": r"[аa]\s*-?\s*98",
    "А-95": r"[аa]\s*-?\s*95",
    "А-92": r"[аa]\s*-?\s*92",
    "Дизель": r"дизел|дизпал|\bдп\b|дп-",
    # Автогаз = скраплений нафтовий газ (суміш пропану й бутану), продається як один товар
    "Газ": r"\bгаз\b|скраплен|пропан|бутан",
}
# «Бензин» без марки не порівнюємо: це може бути і А-92, і А-95, і А-98, а ціни в них різні


def detect_fuel_type(description: str | None, cpv: str | None = None) -> str | None:
    """Вид пального: спершу точний код CPV, якщо він є, інакше — текст опису товару.

    «Дизельне паливо (Євро 5), талон» → «Дизель». Якщо вид не визначено — None.
    """
    cpv = cpv or ""
    if cpv.startswith(NOT_COMPARED_CPV):
        return None
    for prefix, fuel in CPV_FUEL.items():
        if cpv.startswith(prefix):
            return fuel
    text = (description or "").lower()
    found = [fuel for fuel, pattern in FUEL_PATTERNS.items() if re.search(pattern, text)]
    if len(found) != 1:
        return None  # нічого не знайшли або кілька видів в одному рядку з однією ціною — не порівнюємо
    if "бензин" in text and found[0] in ("Дизель", "Газ"):
        return None  # «дизель і бензин» в одному рядку — теж суміш
    return found[0]


def _bounds(prices: pd.Series) -> pd.Series:
    """Межа норми за правилом «ящика з вусами»: Q3 + 1.5 × (Q3 − Q1)."""
    q1, q3 = prices.quantile(0.25), prices.quantile(0.75)
    return pd.Series(q3 + 1.5 * (q3 - q1), index=prices.index)


def _fill_single_fuel_price(items: pd.DataFrame, tenders: pd.DataFrame) -> pd.Series:
    """Якщо в закупівлі один вид пального в кількох рядках (напр. «дизель, талон» + «дизель, налив»)
    і ціни за літр по рядках немає — ділимо суму договору на всі літри.
    Для різних видів пального так робити не можна: середня ціна дизелю й бензину ні з чим не порівнюється."""
    per_tender = items.groupby("id").agg(
        kinds=("fuel_type", lambda s: s.nunique() if s.notna().all() else 0),
        all_liters=("unit", lambda s: (s == "літр").all()),
        no_price=("unit_price", lambda s: s.isna().all()),
        liters=("quantity", "sum"),
    )
    # Однакові рядки без цін (напр. 5 × «дизель, 1 000 000 л») — незрозуміло, чи літри справді
    # сумуються, чи рядок просто продубльовано (по адресах доставки). Таке не ділимо.
    repeated = items[items.duplicated(["id", "description", "quantity"], keep=False)]["id"].unique()
    per_tender["repeated_rows"] = per_tender.index.isin(repeated)
    ok = per_tender[(per_tender["kinds"] == 1) & per_tender["all_liters"] & per_tender["no_price"]
                    & ~per_tender["repeated_rows"]]
    if ok.empty:
        return items["unit_price"]
    money = tenders.set_index("id").loc[ok.index]
    price = money["final_value"] / ok["liters"]
    price = price.where(~money["vat_included"], price / (1 + VAT_RATE))  # усе порівнюємо без ПДВ
    return items["unit_price"].fillna(items["id"].map(price))


def _volume_group(liters: pd.Series) -> pd.Series:
    """Великі закупівлі мають бути дешевшими, тому порівнюємо з такими ж за обсягом."""
    return pd.cut(liters, VOLUME_BINS, labels=VOLUME_LABELS).astype(str)


def _competitive_median(items: pd.DataFrame, keys: list[str]) -> tuple[pd.Series, pd.Series]:
    """Медіана ціни лише серед закупівель з конкуренцією (3+ учасники) і кількість таких закупівель."""
    competitive = items["unit_price"].where(items["bidders"] >= COMPETITIVE_BIDDERS)
    groups = competitive.groupby([items[k] for k in keys])
    return groups.transform("median"), groups.transform("count")


def add_price_stats(items: pd.DataFrame, tenders: pd.DataFrame) -> pd.DataFrame:
    """Для кожного товару: вид пального, справедлива ціна, відхилення і переплата в гривнях.

    Справедлива ціна = медіана ціни за літр у закупівлях з реальною конкуренцією (3+ учасники)
    того самого виду пального і того самого обсягу. Якщо таких мало — беремо ширшу групу:
    усі обсяги, а для пального, де конкурентних закупівель майже немає (газ, А-92), — усі закупівлі.
    """
    items = items.copy()
    items["fuel_type"] = [detect_fuel_type(d, c) for d, c in zip(items["description"], items["cpv"])]
    items["unit_price"] = _fill_single_fuel_price(items, tenders)
    items = items.merge(tenders[["id", "region", "bidders"]], on="id", how="left")
    comparable = (items["unit"] == "літр") & items["unit_price"].between(10, 300) & items["fuel_type"].notna()
    items["unit_price"] = items["unit_price"].where(comparable)
    # Страховка від помилок у даних: ціна вдвічі вища або нижча за медіану свого пального — не віримо
    typical = items.groupby("fuel_type")["unit_price"].transform("median")
    items["unit_price"] = items["unit_price"].where(items["unit_price"].between(typical / 2, typical * 2))
    items["volume_group"] = _volume_group(items["quantity"])

    by_fuel = items.groupby("fuel_type")["unit_price"]
    items["median_price"] = by_fuel.transform("median")
    items["upper_bound"] = by_fuel.transform(_bounds)

    # Справедлива ціна: від найточнішої групи до ширшої, поки аналогів достатньо
    fair_same_volume, n_same_volume = _competitive_median(items, ["fuel_type", "volume_group"])
    fair_fuel, n_fuel = _competitive_median(items, ["fuel_type"])
    fair_all, n_all = by_fuel.transform("median"), by_fuel.transform("count")  # якщо конкурентних мало
    use_volume = n_same_volume >= MIN_ANALOGS
    use_fuel = ~use_volume & (n_fuel >= MIN_ANALOGS)
    use_all = ~use_volume & ~use_fuel & (n_all >= MIN_ANALOGS)
    items["fair_price"] = (fair_same_volume.where(use_volume)
                           .fillna(fair_fuel.where(use_fuel)).fillna(fair_all.where(use_all)))
    items["analogs"] = n_same_volume.where(use_volume).fillna(n_fuel.where(use_fuel)).fillna(n_all.where(use_all))
    items["fair_basis"] = pd.Series(
        [f"з 3+ учасниками, обсяг {v}" if uv else "з 3+ учасниками, усі обсяги" if uf
         else "усіх, бо з 3+ учасниками замало" if ua else None
         for v, uv, uf, ua in zip(items["volume_group"], use_volume, use_fuel, use_all)],
        index=items.index,
    )

    # «Область не вказана» — це замовники з усієї країни, тож з областю їх не порівнюємо
    region_price = items["unit_price"].where(items["region"] != NO_REGION)
    by_region = region_price.groupby([items["fuel_type"], items["region"]])
    items["region_median"] = by_region.transform("median")
    region_enough = by_region.transform("count") >= MIN_REGION_ANALOGS
    items["region_deviation_pct"] = ((items["unit_price"] / items["region_median"] - 1) * 100).where(region_enough)