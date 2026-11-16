"""TenderGuard: Пальне — дашборд Streamlit. Запуск: streamlit run app.py"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from tenderguard import charts
from tenderguard.pipeline import analyze
from tenderguard.prices import analogs_of, no_competition_premium_pct
from tenderguard.rules import MAX_SCORE, RULES

st.set_page_config(page_title="TenderGuard: Пальне", page_icon="⛽", layout="wide")


EMPTY = "—"  # замість None/NaN у таблицях


def as_price_text(values: pd.Series) -> pd.Series:
    """84.5 → «84.50», порожньо → «—» (Streamlit інакше пише в клітинці None)."""
    return values.map(lambda v: EMPTY if pd.isna(v) else f"{v:.2f}")


@st.cache_data
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    result = analyze()
    return result.tenders, result.items


def sidebar_filters(tenders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Фільтри")
    fuels = st.sidebar.multiselect("Вид пального", sorted(items["fuel_type"].dropna().unique()))
    regions = st.sidebar.multiselect("Область", sorted(tenders["region"].dropna().unique()))
    min_score = st.sidebar.slider("Мінімальний Risk Score", 0, MAX_SCORE, 0, step=5)
    query = st.sidebar.text_input("Пошук за замовником або переможцем")

    mask = tenders["risk_score"] >= min_score
    if fuels:
        mask &= tenders["id"].isin(items.loc[items["fuel_type"].isin(fuels), "id"])
    if regions:
        mask &= tenders["region"].isin(regions)
    if query:
        text = tenders["buyer"].fillna("") + " " + tenders["winner"].fillna("")
        mask &= text.str.contains(query, case=False, regex=False)
    return tenders[mask]


def show_metrics(tenders: pd.DataFrame, items: pd.DataFrame) -> None:
    liters = items.loc[items["id"].isin(tenders["id"]) & (items["unit"] == "літр"), "quantity"].sum()
    high = tenders["risk_level"] == "Високий"
    shown_items = items[items["id"].isin(tenders["id"])]
    cols = st.columns(5)
    cols[0].metric("Закупівель пального", f"{len(tenders):,}".replace(",", " "))
    cols[1].metric("Літрів закуплено, млн", f"{liters / 1e6:.1f}")
    cols[2].metric("Переплата понад справедливу ціну, млн грн", f"{tenders['overpay_uah'].sum() / 1e6:.2f}")
    premium = no_competition_premium_pct(shown_items)
    cols[3].metric("Без конкуренції дорожче на", "—" if premium is None else f"{premium:+.1f}%",
                   help="Медіанне відхилення ціни від справедливої в закупівлях з одним учасником")
    cols[4].metric("Високий ризик", int(high.sum()))


def show_top_table(tenders: pd.DataFrame) -> None:
    st.subheader("Закупівлі, які варто перевірити")
    table = tenders[["risk_score", "risk_level", "overpay_uah", "fuel_type", "unit_price", "fair_price",
                     "buyer", "region", "winner", "final_value", "bidders", "url"]].copy()
    table["fuel_type"] = table["fuel_type"].fillna("ціна за літр невідома")
    table["unit_price"] = as_price_text(table["unit_price"])
    table["fair_price"] = as_price_text(table["fair_price"])
    st.dataframe(
        table.head(300), hide_index=True, use_container_width=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn("Бал", min_value=0, max_value=MAX_SCORE, format="%d"),
            "risk_level": "Рівень", "fuel_type": "Пальне",
            "overpay_uah": st.column_config.NumberColumn("Переплата, грн", format="%.0f"),
            "unit_price": "Ціна, грн/л",
            "fair_price": "Справедлива, грн/л",
            "buyer": "Замовник", "region": "Область", "winner": "Переможець",
            "final_value": st.column_config.NumberColumn("Сума, грн", format="%.0f"),
            "bidders": "Учасників",
            "url": st.column_config.LinkColumn("Prozorro", display_text="відкрити"),
        },
    )


def show_tender_card(tenders: pd.DataFrame, items: pd.DataFrame) -> None:
    st.subheader("Чому закупівля потрапила до списку")
    options = tenders.head(300)
    if options.empty:
        st.info("Немає закупівель за цими фільтрами")
        return
    labels = (
        "[" + options["risk_score"].astype(str) + "] " + options["buyer"].fillna("").str.slice(0, 60)
        + " — " + options["title"].fillna("").str.slice(0, 50) + "  #" + options["tender_id"]
    ).tolist()
    row = options.iloc[labels.index(st.selectbox("Оберіть закупівлю", labels))]

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"### {row['risk_score']} / {MAX_SCORE} · {row['risk_level']} ризик")
        st.markdown(
            f"**{row['title']}**  \n"
            f"Замовник: **{row['buyer']}** ({row['region']})  \n"
            f"Переможець: **{row['winner']}**  \n"
            f"Очікувана вартість: {row['expected_value']:,.2f} грн → договір: **{row['final_value']:,.2f} грн**  \n"
            f"Учасників: {row['bidders']} · [відкрити на Prozorro]({row['url']})"
        )
        if row["overpay_uah"] > 0:
            st.error(f"Переплата понад справедливу ціну: ≈ {row['overpay_uah']:,.0f} грн".replace(",", " "))
        for reason in row["reasons"] or ["Жодне правило не спрацювало"]:
            st.warning(reason)
        st.caption("Бал показує кількість ризикових ознак. Це привід перевірити, а не доказ порушення.")
        tender_items = items.loc[items["id"] == row["id"], ["fuel_type", "description", "quantity",
                                                             "unit_price", "fair_price", "region_median"]].copy()
        for column in ("unit_price", "fair_price", "region_median"):
            tender_items[column] = as_price_text(tender_items[column])
        st.dataframe(tender_items, hide_index=True, use_container_width=True, column_config={
            "fuel_type": "Пальне", "description": "Товар", "quantity": "Літрів",
            "unit_price": "Ціна, грн/л",
            "fair_price": "Справедлива ціна",
            "region_median": "Медіана в області",
        })
    with right:
        if pd.notna(row["fair_price"]):
            analogs = analogs_of(items, row["fuel_type"])
            st.plotly_chart(charts.price_comparison(analogs, row), use_container_width=True)
        else:
            st.info("Для цієї закупівлі немає ціни за літр для порівняння")


def show_prices(items: pd.DataFrame) -> None:
    if items.empty:
        st.info("Немає закупівель за цими фільтрами")
        return
    st.plotly_chart(charts.price_boxes(items), use_container_width=True)
    fuel = st.selectbox("Пальне для порівняння областей", sorted(items["fuel_type"].dropna().unique()))
    st.plotly_chart(charts.region_prices(items, fuel), use_container_width=True)


def show_overpay(tenders: pd.DataFrame) -> None:
    st.caption("Справедлива ціна = медіана ціни за літр у закупівлях з реальною конкуренцією (3+ учасники) "
               "того самого пального й обсягу. Переплата = (ціна − справедлива ціна) × літри, "
               "якщо ціна вища більш ніж на 5% (менше — звичайний розкид).")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.overpay_bar(tenders, "buyer", "Замовники, які найбільше переплатили"),
                    use_container_width=True)
    c2.plotly_chart(charts.overpay_bar(tenders, "winner", "Постачальники, яким найбільше переплатили"),
                    use_container_width=True)


def show_rules() -> None:
    with st.expander("Як рахується Risk Score"):
        st.table(pd.DataFrame({"Правило": [r.name for r in RULES], "Бали": [r.points for r in RULES]}))
        st.markdown("**Справедлива ціна:** медіана ціни за літр (без ПДВ) у закупівлях з 3+ учасниками "
                    "того самого виду пального й обсягу (до 2 тис. л, 2–10 тис. л, понад 10 тис. л). "
                    "Бали за ціну ростуть плавно: 0 до +5% над справедливою ціною, максимум від +20%. "
                    "**Межа норми:** Q3 + 1.5 × (Q3 − Q1), правило «ящика з вусами».")


def main() -> None:
    st.title("⛽ TenderGuard: Пальне")
    st.caption("Аналіз ризиків у державних закупівлях пального на відкритих даних Prozorro")

    tenders, items = get_data()
    filtered = sidebar_filters(tenders, items)
    show_metrics(filtered, items)

    tab_list, tab_card, tab_overpay, tab_prices, tab_stats = st.tabs(
        ["Топ ризиків", "Розбір закупівлі", "Переплата", "Ціни на пальне", "Статистика"])
    with tab_list:
        show_top_table(filtered)
    with tab_card:
        show_tender_card(filtered, items)
    with tab_overpay:
        show_overpay(filtered)
    with tab_prices:
        show_prices(items[items["id"].isin(filtered["id"])])
    with tab_stats:
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.score_histogram(filtered), use_container_width=True)
        c2.plotly_chart(charts.flags_bar(filtered), use_container_width=True)
        st.plotly_chart(charts.top_buyers(filtered), use_container_width=True)
    show_rules()
    st.caption(f"Дані: Prozorro Public API (public-api.prozorro.gov.ua), {len(tenders)} завершених закупівель пального")


main()
