"""Tests for full ESPN closing markets (ML, spread, total) and ATS/total push grading."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_betting.data.espn_closing import (
    _assemble_event_markets,
    _extract_provider_mls,
    _pick_best_provider,
    grade_closing_markets,
)


def test_extract_full_markets_with_close_blocks() -> None:
    odds_obj = {
        "provider": {"id": "58", "name": "ESPN BET"},
        "spread": -7.5,
        "overUnder": 216.5,
        "overOdds": -115.0,
        "underOdds": -105.0,
        "open": {
            "total": {"american": "215.0"},
            "over": {"american": "-110"},
            "under": {"american": "-110"},
        },
        "close": {
            "total": {"american": "216.5"},
            "over": {"american": "-115"},
            "under": {"american": "-105"},
        },
        "homeTeamOdds": {
            "favorite": True,
            "moneyLine": -300,
            "spreadOdds": -105.0,
            "open": {
                "pointSpread": {"american": "-6.5"},
                "spread": {"american": "-112"},
                "moneyLine": {"american": "-265"},
            },
            "close": {
                "pointSpread": {"american": "-7.5"},
                "spread": {"american": "-105"},
                "moneyLine": {"american": "-300"},
            },
        },
        "awayTeamOdds": {
            "favorite": False,
            "moneyLine": 240,
            "spreadOdds": -115.0,
            "open": {
                "pointSpread": {"american": "+6.5"},
                "spread": {"american": "-108"},
                "moneyLine": {"american": "+215"},
            },
            "close": {
                "pointSpread": {"american": "+7.5"},
                "spread": {"american": "-115"},
                "moneyLine": {"american": "+240"},
            },
        },
    }
    parsed = _extract_provider_mls(odds_obj)
    assert parsed["provider_id"] == "58"
    assert parsed["has_close_block"] is True
    assert parsed["has_open_block"] is True
    assert parsed["home_ml"] == -300.0
    assert parsed["away_ml"] == 240.0
    assert parsed["open_home_ml"] == -265.0
    assert parsed["open_away_ml"] == 215.0
    assert parsed["spread_home"] == -7.5
    assert parsed["spread_home_odds"] == -105.0
    assert parsed["spread_away_odds"] == -115.0
    assert parsed["open_spread_home"] == -6.5
    assert parsed["open_spread_home_odds"] == -112.0
    assert parsed["open_spread_away_odds"] == -108.0
    assert parsed["total_line"] == 216.5
    assert parsed["over_odds"] == -115.0
    assert parsed["under_odds"] == -105.0
    assert parsed["open_total_line"] == 215.0
    assert parsed["open_over_odds"] == -110.0
    assert parsed["open_under_odds"] == -110.0

    assembled = _assemble_event_markets("401704699", [parsed])
    assert abs(assembled["p_home_shin"] + assembled["p_away_shin"] - 1.0) < 1e-9
    # Shin penalizes heavy longshots more than multiplicative devig -> favorite Shin prob > novig prob
    assert assembled["p_home_shin"] > assembled["p_home_novig"]


def test_even_and_pk_parsing() -> None:
    odds_obj = {
        "provider": {"id": "100", "name": "Draft Kings"},
        "close": {
            "total": {"american": "228"},
            "over": {"american": "EVEN"},
            "under": {"american": "-120"},
        },
        "homeTeamOdds": {
            "close": {
                "pointSpread": {"american": "PK"},
                "spread": {"american": "-110"},
                "moneyLine": {"american": "-110"},
            }
        },
        "awayTeamOdds": {
            "close": {
                "pointSpread": {"american": "PK"},
                "spread": {"american": "-110"},
                "moneyLine": {"american": "EVEN"},
            }
        },
    }
    parsed = _extract_provider_mls(odds_obj)
    assert parsed["spread_home"] == 0.0
    assert parsed["away_ml"] == 100.0
    assert parsed["over_odds"] == 100.0
    assert parsed["under_odds"] == -120.0
    assert parsed["total_line"] == 228.0


def test_live_and_projection_providers_strictly_rejected() -> None:
    live_only = [
        {
            "provider_id": "59",
            "provider_name": "ESPN Bet - Live Odds",
            "home_ml": -5000,
            "away_ml": 1800,
            "spread_home": -18.5,
            "spread_home_odds": -115.0,
            "spread_away_odds": -115.0,
            "total_line": 244.5,
            "over_odds": -110.0,
            "under_odds": -110.0,
        },
        {
            "provider_id": "1001",
            "provider_name": "accuscore",
            "home_ml": -150,
            "away_ml": 130,
            "spread_home": -3.5,
            "spread_home_odds": -110.0,
            "spread_away_odds": -110.0,
            "total_line": 220.5,
            "over_odds": -110.0,
            "under_odds": -110.0,
        },
    ]
    assert _pick_best_provider(live_only, market="ml") is None
    assert _pick_best_provider(live_only, market="spread") is None
    assert _pick_best_provider(live_only, market="total") is None


def test_corrupted_spread_sign_vs_strong_ml_rejected_and_falls_back() -> None:
    bad_unibet = _extract_provider_mls(
        {
            "provider": {"id": "36", "name": "Unibet"},
            "spread": 1.5,  # contradicts -165 home ML favorite
            "overUnder": 215.5,
            "overOdds": -110.0,
            "underOdds": -110.0,
            "homeTeamOdds": {"moneyLine": -165, "spreadOdds": -186.0},
            "awayTeamOdds": {"moneyLine": 138, "spreadOdds": 150.0},
        }
    )
    assert bad_unibet["spread_home"] is None

    clean_dk = _extract_provider_mls(
        {
            "provider": {"id": "40", "name": "DraftKings"},
            "spread": -4.0,
            "overUnder": 214.0,
            "overOdds": -110.0,
            "underOdds": -110.0,
            "homeTeamOdds": {"moneyLine": -175, "spreadOdds": -110.0},
            "awayTeamOdds": {"moneyLine": 150, "spreadOdds": -110.0},
        }
    )
    assembled = _assemble_event_markets("401468228", [bad_unibet, clean_dk])
    assert assembled["spread_home"] == -4.0
    assert assembled["spread_provider"] == "DraftKings"


def test_grade_closing_markets_handles_ats_and_total_pushes() -> None:
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "home_score": [112.0, 110.0, 105.0],
            "away_score": [106.0, 104.0, 115.0],
            "spread_home": [-5.5, -6.0, 4.5],
            "total_line": [217.5, 214.0, 225.0],
        }
    )
    g = grade_closing_markets(df)

    # g1: margin +6 vs -5.5 -> home covers (+0.5); total 218 vs 217.5 -> over
    assert g.loc[0, "home_cover"] == 1.0
    assert bool(g.loc[0, "spread_push"]) is False
    assert g.loc[0, "ats_result"] == "home"
    assert g.loc[0, "over_hit"] == 1.0
    assert bool(g.loc[0, "total_push"]) is False
    assert g.loc[0, "total_result"] == "over"

    # g2: margin +6 vs -6.0 -> exact ATS push; total 214 vs 214.0 -> exact total push
    assert bool(g.loc[1, "spread_push"]) is True
    assert np.isnan(g.loc[1, "home_cover"])
    assert np.isnan(g.loc[1, "away_cover"])
    assert g.loc[1, "ats_result"] == "push"
    assert bool(g.loc[1, "total_push"]) is True
    assert np.isnan(g.loc[1, "over_hit"])
    assert np.isnan(g.loc[1, "under_hit"])
    assert g.loc[1, "total_result"] == "push"

    # g3: margin -10 + 4.5 = -5.5 -> away covers; total 220 vs 225 -> under
    assert g.loc[2, "home_cover"] == 0.0
    assert g.loc[2, "away_cover"] == 1.0
    assert g.loc[2, "ats_result"] == "away"
    assert g.loc[2, "over_hit"] == 0.0
    assert g.loc[2, "under_hit"] == 1.0
    assert g.loc[2, "total_result"] == "under"
