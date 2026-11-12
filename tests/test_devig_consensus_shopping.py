"""Unit tests for devigging, sharp consensus, stale-outlier detection, and cross-book routing."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from nba_betting.cli import app
from nba_betting.data.odds import fetch_nba_odds, parse_all_book_odds, parse_event_odds, pull_odds
from nba_betting.pricing.american import devig_two_way
from nba_betting.pricing.consensus import (
    BOOK_SHARPNESS_WEIGHTS,
    TwoWayQuote,
    compute_sharp_consensus,
    detect_stale_outliers,
    is_valid_pregame_book,
    normalize_book_key,
)
from nba_betting.pricing.devig import (
    SUPPORTED_DEVIG_METHODS,
    compare_devig_methods,
    devig_american,
    devig_decimal,
    devig_probabilities,
    power_exponent,
    probit_shift,
    shin_insider_fraction,
)
from nba_betting.pricing.edge import build_edge_rows
from nba_betting.pricing.shopping import (
    analyze_slate_markets,
    load_multi_book_file,
    parse_live_slate_markdown,
    route_parlays_across_books,
    route_single_bets_across_books,
)


@pytest.mark.parametrize("method", SUPPORTED_DEVIG_METHODS)
def test_symmetric_even_market_returns_half_for_all_methods(method: str) -> None:
    p_a, p_b = devig_american((-110.0, -110.0), method=method)
    assert p_a == pytest.approx(0.5, abs=1e-12)
    assert p_b == pytest.approx(0.5, abs=1e-12)
    p2_a, p2_b = devig_two_way(-110.0, -110.0, method=method)
    assert p2_a == pytest.approx(0.5, abs=1e-12)
    assert p2_b == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize(
    "prices",
    [
        (-150.0, 130.0),
        (-260.0, 210.0),
        (-500.0, 375.0),
        (-1100.0, 700.0),
        (-3000.0, 1200.0),
    ],
)
def test_power_shin_probit_penalize_longshot_vs_multiplicative(
    prices: tuple[float, float],
) -> None:
    """Core FLB theorem: Power, Shin, and Probit assign strictly lower fair probability
    to the longshot (and higher to the favorite) than Multiplicative devig when S > 1.
    """
    res = compare_devig_methods(prices)
    for m in SUPPORTED_DEVIG_METHODS:
        p_fav, p_dog = res[m]
        assert 0.0 < p_dog < p_fav < 1.0
        assert p_fav + p_dog == pytest.approx(1.0, abs=1e-11)

    _, dog_mult = res["multiplicative"]
    _, dog_shin = res["shin"]
    _, dog_probit = res["probit"]
    _, dog_power = res["power"]
    _, dog_add = res["additive"]

    assert dog_shin < dog_mult - 1e-5
    assert dog_probit < dog_mult - 1e-5
    assert dog_power < dog_mult - 1e-5
    assert dog_add < dog_mult - 1e-5


def test_longshot_penalty_increases_with_extremity() -> None:
    """Relative haircut on the underdog grows as the underdog becomes a longer shot."""
    mild = compare_devig_methods((-140.0, 120.0))
    heavy = compare_devig_methods((-1000.0, 650.0))

    mild_ratio_power = mild["power"][1] / mild["multiplicative"][1]
    heavy_ratio_power = heavy["power"][1] / heavy["multiplicative"][1]
    assert heavy_ratio_power < mild_ratio_power

    mild_ratio_shin = mild["shin"][1] / mild["multiplicative"][1]
    heavy_ratio_shin = heavy["shin"][1] / heavy["multiplicative"][1]
    assert heavy_ratio_shin < mild_ratio_shin


def test_multiway_three_outcome_devigging() -> None:
    """All five methods work on N >= 3 outcomes and penalize the 3-way longshot."""
    # Example 3-way market (e.g. regulation 1X2 or 3-way margin band): S = 1.07
    implied = (0.70, 0.25, 0.12)
    mult = devig_probabilities(implied, method="multiplicative")
    shin = devig_probabilities(implied, method="shin")
    power = devig_probabilities(implied, method="power")
    probit = devig_probabilities(implied, method="probit")
    add = devig_probabilities(implied, method="additive")

    for vec in (mult, shin, power, probit, add):
        assert len(vec) == 3
        assert sum(vec) == pytest.approx(1.0, abs=1e-11)
        assert all(0.0 < p < 1.0 for p in vec)

    # Longshot (0.12 raw) must be discounted by Shin, Power, Probit, Additive vs Multiplicative
    assert shin[2] < mult[2]
    assert power[2] < mult[2]
    assert probit[2] < mult[2]
    assert add[2] < mult[2]
    # Note for N=3, Shin differs strictly from Additive!
    assert abs(shin[2] - add[2]) > 1e-3

    assert power_exponent(implied) > 1.0
    assert 0.0 < shin_insider_fraction(implied) < 1.0
    assert probit_shift(implied) > 0.0


def test_zero_vig_and_synthetic_arb_markets() -> None:
    zero_vig = (0.60, 0.40)
    for m in SUPPORTED_DEVIG_METHODS:
        p = devig_probabilities(zero_vig, method=m)
        assert p == pytest.approx((0.60, 0.40), abs=1e-10)

    # Cross-book synthetic arb: +120 and +105 => implied sum = 0.4545 + 0.4878 = 0.9423 < 1
    arb_decs = (2.20, 2.05)
    for m in SUPPORTED_DEVIG_METHODS:
        p = devig_decimal(arb_decs, method=m)
        assert sum(p) == pytest.approx(1.0, abs=1e-10)
        assert 0.0 < p[0] < p[1] < 1.0


def test_sharp_consensus_weights_lowvig_and_pinnacle_over_soft_books() -> None:
    assert BOOK_SHARPNESS_WEIGHTS["pinnacle"] > BOOK_SHARPNESS_WEIGHTS["lowvig"]
    assert BOOK_SHARPNESS_WEIGHTS["lowvig"] > BOOK_SHARPNESS_WEIGHTS["fanduel"]
    assert BOOK_SHARPNESS_WEIGHTS["betonlineag"] > BOOK_SHARPNESS_WEIGHTS["betmgm"]

    # Sharp books price Home fair at 60% (-155 / +142, ~2.3% hold);
    # three soft high-hold books shade Home to 50% (-115 / -115, ~7% hold).
    quotes = [
        TwoWayQuote("Pinnacle", -155, 142, market="moneyline"),
        TwoWayQuote("LowVig.ag", -156, 142, market="moneyline"),
        TwoWayQuote("BetOnline.ag", -158, 140, market="moneyline"),
        TwoWayQuote("MyBookie.ag", -115, -115, market="moneyline"),
        TwoWayQuote("BetUS", -115, -115, market="moneyline"),
        TwoWayQuote("Bovada", -115, -115, market="moneyline"),
    ]
    cons = compute_sharp_consensus(quotes, devig_method="shin")
    # Equal-weighted mean would be ~0.55; sharp + inverse-hold consensus stays close to 0.60!
    assert cons.prob_a > 0.585
    assert cons.best_book_a in {"MyBookie.ag", "BetUS", "Bovada"}
    assert cons.best_price_a == -115.0

    # LOO outlier detector flags the three stale -115 books as +EV on side A!
    outliers = detect_stale_outliers(quotes, devig_method="shin", min_ev=0.02, min_peer_weight=4.0)
    flagged_books = {o.canonical_book for o in outliers if o.side == "a"}
    assert {"mybookieag", "betus", "bovada"} <= flagged_books
    for o in outliers:
        assert o.ev_per_unit > 0.05


def test_state_clone_deduplication_and_live_book_filtering() -> None:
    assert not is_valid_pregame_book("ESPN Bet - Live Odds", "59")
    assert not is_valid_pregame_book("Caesars Sportsbook (New Jersey) - Live Odds", "46")
    assert not is_valid_pregame_book("consensus", "1004")
    assert not is_valid_pregame_book("teamrankings", "1002")
    assert is_valid_pregame_book("LowVig.ag")
    assert normalize_book_key("Caesars Sportsbook (Colorado)") == "caesars"
    assert normalize_book_key("williamhill_us") == "caesars"

    # Three Caesars state clones at +200/-250 must not overwhelm LowVig at -110/-110
    quotes = [
        TwoWayQuote("LowVig.ag", -105, -105, market="moneyline"),
        TwoWayQuote("Caesars Sportsbook (Colorado)", -200, 170, market="moneyline"),
        TwoWayQuote("Caesars Sportsbook (New Jersey)", -200, 170, market="moneyline"),
        TwoWayQuote("Caesars Sportsbook (Tennessee)", -200, 170, market="moneyline"),
    ]
    cons = compute_sharp_consensus(quotes, deduplicate_canonical_books=True)
    assert cons.n_books == 2
    assert (
        cons.prob_a < 0.53
    )  # LowVig (w=5, low hold) dominates single deduplicated Caesars (w=0.75)


def test_point_adjusted_spread_stale_number_detection() -> None:
    """Unjuiced stale spread is flagged +EV; heavily juiced half-point is rejected."""
    quotes = [
        TwoWayQuote("LowVig.ag", -111, -101, line_a=-6.0, market="spread"),
        TwoWayQuote("BetOnline.ag", -115, -105, line_a=-6.0, market="spread"),
        TwoWayQuote("DraftKings", -108, -112, line_a=-6.5, market="spread"),
        TwoWayQuote("FanDuel", -110, -110, line_a=-6.5, market="spread"),
        TwoWayQuote("BetMGM", -110, -110, line_a=-6.5, market="spread"),
        TwoWayQuote("BetRivers", -108, -112, line_a=-5.0, market="spread"),
    ]
    outliers = detect_stale_outliers(quotes, devig_method="shin", min_ev=0.0, min_peer_weight=4.0)
    assert outliers
    top = outliers[0]
    assert top.canonical_book == "betrivers"
    assert top.side == "a"
    assert top.book_line == -5.0
    assert top.line_out_pts is not None and top.line_out_pts >= 1.0
    assert top.ev_per_unit > 0.01

    # When the -5.5 book charges -122 juice, the vig eats the half-point — must NOT false-positive!
    juiced_quotes = quotes[:-1] + [
        TwoWayQuote("BetRivers", -122, 100, line_a=-5.5, market="spread")
    ]
    juiced_outliers = [
        o
        for o in detect_stale_outliers(
            juiced_quotes, devig_method="shin", min_ev=0.0, min_peer_weight=4.0
        )
        if o.canonical_book == "betrivers" and o.side == "a"
    ]
    assert juiced_outliers == []


def test_odds_py_retains_all_books_and_builds_multi_book_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test for the odds.py books[0] bug: all 9 books must be retained."""
    multi_book_payload = [
        {
            "id": "evt_orl_sac",
            "sport_key": "basketball_nba",
            "commence_time": "2026-11-11T03:10:00Z",
            "home_team": "Sacramento Kings",
            "away_team": "Orlando Magic",
            "bookmakers": [
                {
                    "key": "betmgm",
                    "title": "BetMGM",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Orlando Magic", "price": -235},
                                {"name": "Sacramento Kings", "price": 190},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Orlando Magic", "price": -105, "point": -6.5},
                                {"name": "Sacramento Kings", "price": -115, "point": 6.5},
                            ],
                        },
                    ],
                },
                {
                    "key": "lowvig",
                    "title": "LowVig.ag",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Orlando Magic", "price": -228},
                                {"name": "Sacramento Kings", "price": 195},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Orlando Magic", "price": -111, "point": -6.0},
                                {"name": "Sacramento Kings", "price": -101, "point": 6.0},
                            ],
                        },
                    ],
                },
                {
                    "key": "betonlineag",
                    "title": "BetOnline.ag",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Orlando Magic", "price": -230},
                                {"name": "Sacramento Kings", "price": 192},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Orlando Magic", "price": -115, "point": -6.0},
                                {"name": "Sacramento Kings", "price": -105, "point": 6.0},
                            ],
                        },
                    ],
                },
                {
                    "key": "betrivers",
                    "title": "BetRivers",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Orlando Magic", "price": -225},
                                {"name": "Sacramento Kings", "price": 180},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Orlando Magic", "price": -114, "point": -5.5},
                                {"name": "Sacramento Kings", "price": -108, "point": 5.5},
                            ],
                        },
                    ],
                },
            ],
        }
    ]
    monkeypatch.setenv("ODDS_API_KEY", "dummy-key")
    all_rows = parse_all_book_odds(multi_book_payload[0])
    assert len(all_rows) == 4
    assert {r.book for r in all_rows} == {"betmgm", "lowvig", "betonlineag", "betrivers"}

    # Legacy single-book helper still works when requested
    one = parse_event_odds(multi_book_payload[0], book_key="lowvig")
    assert one is not None and one.book == "lowvig"

    # fetch_nba_odds & pull_odds retain ALL books by default
    fetched = fetch_nba_odds(http_get=lambda u, p: multi_book_payload)
    assert len(fetched) == 4
    cache_path = pull_odds(root=tmp_path, http_get=lambda u, p: multi_book_payload)
    reloaded = load_multi_book_file(cache_path)
    assert len(reloaded) == 1
    assert len(reloaded[0].books) == 4

    # build_edge_rows emits edges across ALL books, not just books[0]
    from nba_betting.models.game import GamePrediction

    pred = GamePrediction(
        game_id="g1",
        game_date="2026-11-10",
        home_tricode="SAC",
        away_tricode="ORL",
        exp_home_score=109.0,
        exp_away_score=115.5,
        exp_total=224.5,
        exp_margin=-6.5,
        p_home_win=0.31,
        p_away_win=0.69,
        p_home_cover=0.50,
        p_over=0.50,
        fair_spread=6.5,
        fair_total=224.5,
        market_spread=6.5,
        market_total=224.5,
        n_sims=2000,
    )
    edge_rows = build_edge_rows([pred], fetched, n_sims=2000)
    books_in_edges = {r.book for r in edge_rows}
    assert books_in_edges == {"betmgm", "lowvig", "betonlineag", "betrivers"}


def test_parse_captured_live_slate_and_route_singles_and_parlays() -> None:
    slate_md = Path("/cursor/stores/bc-1f2d201a-82aa-4cd4-87c5-88f04e20e03c/docs/nba-live-slate.md")
    if not slate_md.exists():
        pytest.skip("shared store nba-live-slate.md not mounted")

    events = parse_live_slate_markdown(slate_md)
    assert len(events) == 10
    nov10 = [e for e in events if len(e.books) == 9]
    dec25 = [e for e in events if len(e.books) == 1]
    assert len(nov10) == 6
    assert len(dec25) == 4

    cons_rows, outlier_rows = analyze_slate_markets(
        events, devig_method="shin", min_peer_weight=4.0
    )
    assert len(cons_rows) == 18  # 6 Nov-10 games * 3 markets; 1-book Dec-25 games excluded!
    assert len(outlier_rows) > 0

    singles = route_single_bets_across_books(events, devig_method="shin", bankroll=50.0)
    parlays = route_parlays_across_books(events, devig_method="shin", sizes=(2, 3), bankroll=50.0)
    assert len(singles) > 0
    assert len(parlays) > 0
    for t in parlays:
        assert t.routed_ev_per_unit > 0.0
        assert t.routed_ev_per_unit >= t.median_book_ev
        assert len({leg.event_id for leg in t.legs}) == t.n_legs

    # Test CLI slate shop command end-to-end
    runner = CliRunner()
    res = runner.invoke(app, ["slate", "shop", "--odds-file", str(slate_md), "--bankroll", "50"])
    assert res.exit_code == 0
    assert "STALE OUTLIER QUOTES" in res.stdout
    assert "BEST CROSS-BOOK ROUTED PARLAYS" in res.stdout
