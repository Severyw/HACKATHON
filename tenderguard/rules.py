from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

PRICE_START_PCT = 5
PRICE_FULL_PCT = 20


@dataclass(frozen=True)
class Rule:
    name: str
    points: int
    strength: Callable[[pd.DataFrame], pd.Series]
    explain: Callable[[pd.Series], str]


def _uah(amount: float) -> str:
    return f"{amount:,.0f}".replace(",", " ")


def _repeat_wins(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["buyer_id", "winner_id"])["id"].transform("count")


def _price_strength(df: pd.DataFrame) -> pd.Series:
    span = PRICE_FULL_PCT - PRICE_START_PCT
    return ((df["price_deviation_pct"] - PRICE_START_PCT) / span).clip(0, 1).fillna(0)


RULES = [
    Rule(
        name="Ціна вища за справедливу",
        points=40,
        strength=_price_strength,
        explain=lambda r: (
            f"{r['fuel_type']} по {r['unit_price']:.2f} грн/л: на {r['price_deviation_pct']:.0f}% дорожче "
            f"за справедливу ціну {r['fair_price']:.2f} грн/л (медіана {int(r['analogs'])} закупівель: "
            f"{r['fair_basis']}). Переплата ≈ {_uah(r['overpay_uah'])} грн"
        ),
    ),
    Rule(
        name="Один учасник",
        points=25,
        strength=lambda df: df["bidders"] == 1,
        explain=lambda r: "Подано лише одну пропозицію, конкуренції не було",
    ),
    Rule(
        name="Імітація конкуренції",
        points=15,
        strength=lambda df: (df["bidders"] >= 2) & (df["savings_pct"] < 1),
        explain=lambda r: (f"Учасників {r['bidders']}, але ціна впала лише на {r['savings_pct']:.1f}% "
                           f"від очікуваної: схоже, що реально не змагалися"),
    ),
    Rule(
        name="Постійний переможець",
        points=10,
        strength=lambda df: _repeat_wins(df) >= 3,
        explain=lambda r: f"Цей постачальник виграв у замовника {int(r['repeat_wins'])} разів",
    ),
    Rule(
        name="Дорожче, ніж в області",
        points=10,
        strength=lambda df: df["region_deviation_pct"] >= 10,
        explain=lambda r: (f"На {r['region_deviation_pct']:.0f}% дорожче за медіану області "
                           f"({r['region_median']:.2f} грн/л, {r['region']})"),
    ),
]

MAX_SCORE = sum(rule.points for rule in RULES)


def risk_level(score: int) -> str:
    if score >= 60:
        return "Високий"
    if score >= 35:
        return "Середній"
    return "Низький"


def apply_rules(df: pd.DataFrame, rules: list[Rule] = RULES) -> pd.DataFrame:
    df = df.copy()
    df["repeat_wins"] = _repeat_wins(df)

    for rule in rules:
        df[rule.name] = rule.strength(df).fillna(0).astype(float)

    df["risk_score"] = sum(df[rule.name] * rule.points for rule in rules).round().astype(int)
    df["risk_level"] = df["risk_score"].map(risk_level)
    df["reasons"] = df.apply(lambda row: explain_row(row, rules), axis=1)
    return df


def explain_row(row: pd.Series, rules: list[Rule] = RULES) -> list[str]:
    return [f"+{round(row[rule.name] * rule.points)}  {rule.explain(row)}"
            for rule in rules if row[rule.name] > 0]
