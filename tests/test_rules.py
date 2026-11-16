import pandas as pd

from tenderguard.prices import add_price_stats, detect_fuel_type, no_competition_premium_pct, summarize_by_tender
from tenderguard.rules import MAX_SCORE, apply_rules


def test_fuel_type_from_description():
    assert detect_fuel_type("Дизельне паливо (Євро 5), талон") == "Дизель"
    assert detect_fuel_type("Бензин А-95 (Євро 5), картка") == "А-95"
    assert detect_fuel_type("Газ нафтовий скраплений") == "Газ"
    assert detect_fuel_type("Канцтовари") is None
    assert detect_fuel_type("Паливо для газотурбінних двигунів Джет А-1") is None  # не «газ»


def test_mixed_fuels_in_one_line_are_not_compared():
    # одна ціна на бензин і дизель разом — порівнювати з медіаною А-95 нечесно
    assert detect_fuel_type("Бензин А-95 (талони) та дизельне паливо Євро 5 (талони)") is None
    assert detect_fuel_type("Бензин автомобільний А-95") == "А-95"
    assert detect_fuel_type("Дизельне паливо, бензин") is None


def test_exact_cpv_wins_over_text():
    assert detect_fuel_type("Паливо", cpv="09134200-9") == "Дизель"
    assert detect_fuel_type("Бензин А-95", cpv="09132000-3") == "А-95"


def test_gasoline_without_grade_is_not_compared():
    # А-92, А-95 і А-98 коштують по-різному, тож «просто бензин» не порівнюємо
    assert detect_fuel_type("Бензин") is None
    assert detect_fuel_type("Бензин", cpv="09132000-3") is None


def make_data(prices, **tender_overrides):
    """Закупівлі дизелю по 1000 л з заданими цінами за літр; остання — та, що перевіряємо."""
    n = len(prices)
    tenders = pd.DataFrame(dict(
        id=[f"t{i}" for i in range(n)], region=["Київська область"] * n, buyer_id=["b"] * n,
        winner_id=[f"w{i}" for i in range(n)], bidders=[3] * n, savings_pct=[5.0] * n,
    ))
    for column, value in tender_overrides.items():
        tenders[column] = value
    items = pd.DataFrame(dict(id=tenders["id"], description="Дизельне паливо", cpv="09130000-9",
                              unit="літр", quantity=1000, unit_price=prices))
    items = add_price_stats(items, tenders)
    tenders = tenders.merge(summarize_by_tender(items), on="id", how="left")
    return apply_rules(tenders)


NORMAL = [84.0, 85.0, 85.5, 86.0, 84.5, 85.2, 86.5, 83.9, 85.8, 84.7]  # справедлива ціна ≈ 85.1


def test_normal_diesel_has_zero_score():
    df = make_data(NORMAL + [85.0])
    assert df["risk_score"].iloc[-1] == 0
    assert df["reasons"].iloc[-1] == []


def test_overpay_is_counted_in_uah():
    df = make_data(NORMAL + [95.2])
    # (95.2 − ~85.2 справедлива ціна) × 1000 л ≈ 10 000 грн
    assert 9_000 < df["overpay_uah"].iloc[-1] < 11_000


def test_price_points_grow_with_overpay():
    small = make_data(NORMAL + [92.0])["risk_score"].iloc[-1]   # ≈ +8%
    big = make_data(NORMAL + [105.0])["risk_score"].iloc[-1]    # ≈ +23%
    assert 0 < small < big


def test_single_bidder_overpriced_repeat_winner():
    df = make_data(NORMAL + [120.0], winner_id="same")
    df.loc[df.index[-1], "bidders"] = 1  # перевіряємо лише останню закупівлю
    df = apply_rules(df)
    # з одним учасником «імітація конкуренції» неможлива, тож максимум — без її 15 балів
    assert df["risk_score"].iloc[-1] == MAX_SCORE - 15


def test_imitated_competition():
    df = make_data(NORMAL + [85.0], bidders=4, savings_pct=0.2)
    assert df["Імітація конкуренції"].iloc[-1] == 1


def test_fair_price_comes_from_competitive_tenders():
    """Закупівлі з одним учасником дорожчі, але справедлива ціна — з тих, де була конкуренція."""
    tenders = pd.DataFrame(dict(id=[f"t{i}" for i in range(20)], region="Київська область",
                                bidders=[3] * 10 + [1] * 10))
    items = pd.DataFrame(dict(id=tenders["id"], description="Дизельне паливо", cpv="09130000-9",
                              unit="літр", quantity=1000, unit_price=[83.0] * 10 + [90.0] * 10))
    items = add_price_stats(items, tenders)
    assert items["fair_price"].iloc[0] == 83.0
    assert round(no_competition_premium_pct(items), 1) == round((90 / 83 - 1) * 100, 1)


def test_one_fuel_in_several_lines_gets_price_from_total():
    """«Дизель, талон» + «дизель, налив» без цін по рядках: сума / усі літри."""
    tenders = pd.DataFrame(dict(id=["t0"], region="Київська область", bidders=3,
                                final_value=170_000.0, vat_included=False))
    items = pd.DataFrame(dict(id=["t0", "t0"], description=["Дизельне паливо, талон", "Дизельне паливо, налив"],
                              cpv="09130000-9", unit="літр", quantity=[1000, 1000], unit_price=[None, None]))
    assert list(add_price_stats(items, tenders)["unit_price"]) == [85.0, 85.0]


def test_different_fuels_with_one_total_are_not_divided():
    tenders = pd.DataFrame(dict(id=["t0"], region="Київська область", bidders=3,
                                final_value=157_000.0, vat_included=False))
    items = pd.DataFrame(dict(id=["t0", "t0"], description=["Дизельне паливо", "Бензин А-95"],
                              cpv="09130000-9", unit="літр", quantity=[1000, 1000], unit_price=[None, None]))
    assert add_price_stats(items, tenders)["unit_price"].isna().all()


def test_unknown_region_is_not_compared_as_a_region():
    """«Область не вказана» — замовники з усієї країни, порівнювати з ними як з областю не можна."""
    tenders = pd.DataFrame(dict(id=[f"t{i}" for i in range(6)], region="Область не вказана", bidders=3))
    items = pd.DataFrame(dict(id=tenders["id"], description="Дизельне паливо", cpv="09130000-9",
                              unit="літр", quantity=1000, unit_price=[85.0] * 5 + [100.0]))
    assert add_price_stats(items, tenders)["region_deviation_pct"].isna().all()


def test_repeated_rows_without_prices_are_not_divided():
    """5 однакових рядків «дизель, 1 000 000 л» і одна сума: незрозуміло, скільки літрів насправді."""
    tenders = pd.DataFrame(dict(id=["t0"], region="Київська область", bidders=3,
                                final_value=89_000_000.0, vat_included=False))
    items = pd.DataFrame(dict(id=["t0"] * 5, description="Дизельне паливо", cpv="09130000-9",
                              unit="літр", quantity=1_000_000, unit_price=[None] * 5))
    assert add_price_stats(items, tenders)["unit_price"].isna().all()
