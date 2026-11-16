from __future__ import annotations

import pandas as pd
import plotly.express as px

from tenderguard.loader import NO_REGION
from tenderguard.rules import RULES

LEVEL_COLORS = {"Низький": "#4caf50", "Середній": "#ff9800", "Високий": "#e53935"}


def score_histogram(df: pd.DataFrame):
    fig = px.histogram(
        df, x="risk_score", color="risk_level", nbins=20, color_discrete_map=LEVEL_COLORS,
        labels={"risk_score": "Risk Score", "risk_level": "Рівень"},
        title="Розподіл закупівель за Risk Score",
    )
    return fig.update_layout(yaxis_title="Кількість закупівель", bargap=0.05)


def flags_bar(df: pd.DataFrame):
    counts = pd.Series({rule.name: int((df[rule.name] > 0).sum()) for rule in RULES}).sort_values()
    return px.bar(x=counts.values, y=counts.index, orientation="h",
                  labels={"x": "Закупівель", "y": ""}, title="Скільки разів спрацювало кожне правило")


def price_boxes(items: pd.DataFrame):
    data = items.dropna(subset=["unit_price"])
    fig = px.box(data, x="fuel_type", y="unit_price", points="outliers", color="fuel_type",
                 labels={"fuel_type": "", "unit_price": "Ціна, грн/л без ПДВ"},
                 title="Ціна за літр за видами пального")
    return fig.update_layout(showlegend=False)


def region_prices(items: pd.DataFrame, fuel_type: str):
    data = (items[(items["fuel_type"] == fuel_type) & (items["region"] != NO_REGION)]
            .dropna(subset=["unit_price", "region"])
            .groupby("region")["unit_price"].agg(["median", "count"]).query("count >= 3")
            .sort_values("median"))
    fig = px.bar(data, x="median", y=data.index, orientation="h", hover_data={"count": True},
                 labels={"median": "Медіана, грн/л", "y": "", "count": "Закупівель"},
                 title=f"{fuel_type}: медіанна ціна за областями")
    return fig.update_layout(yaxis_title="", height=max(350, 22 * len(data)))


def top_buyers(df: pd.DataFrame, n: int = 10, min_tenders: int = 2):
    stats = (df.groupby("buyer")
             .agg(tenders=("id", "count"), avg_score=("risk_score", "mean"), total=("final_value", "sum"))
             .query("tenders >= @min_tenders").nlargest(n, "avg_score").sort_values("avg_score"))
    fig = px.bar(stats, x="avg_score", y=stats.index.str.slice(0, 60), orientation="h",
                 hover_data={"tenders": True, "total": ":,.0f"},
                 labels={"avg_score": "Середній Risk Score", "y": "", "tenders": "Закупівель", "total": "Сума, грн"},
                 title=f"Замовники з найвищим середнім балом (від {min_tenders} закупівель)")
    return fig.update_layout(yaxis_title="")


def price_comparison(analogs: pd.DataFrame, row: pd.Series):
    data = analogs.sort_values("unit_price").reset_index(drop=True)
    data["who"] = data["id"].eq(row["id"]).map({True: "Ця закупівля", False: "Інші закупівлі"})
    fig = px.bar(data, x=data.index, y="unit_price", color="who",
                 color_discrete_map={"Ця закупівля": "#e53935", "Інші закупівлі": "#90a4ae"},
                 hover_data={"region": True, "quantity": ":,.0f", "unit_price": ":.2f"},
                 labels={"unit_price": "Ціна, грн/л", "x": "", "who": "", "region": "Область",
                         "quantity": "Літрів"},
                 title=f"{row['fuel_type']}: ціна цієї закупівлі серед {len(data) - 1} інших")
    fig.add_hline(y=row["fair_price"], line_dash="dash",
                  annotation_text=f"справедлива ціна ({row['fair_basis']})")
    return fig.update_xaxes(showticklabels=False)


def overpay_bar(df: pd.DataFrame, by: str, title: str, n: int = 10):
    stats = (df.groupby(by).agg(overpay=("overpay_uah", "sum"), tenders=("id", "count"))
             .query("overpay > 0").nlargest(n, "overpay").sort_values("overpay"))
    fig = px.bar(stats, x="overpay", y=stats.index.str.slice(0, 60), orientation="h",
                 hover_data={"tenders": True}, color_discrete_sequence=["#e53935"],
                 labels={"overpay": "Переплата, грн", "y": "", "tenders": "Закупівель"}, title=title)
    return fig.update_layout(yaxis_title="")
