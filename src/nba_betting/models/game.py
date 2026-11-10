"""Simple game score engine: pace + efficiency → (home, away) distribution.

Expected possessions from average team pace; expected points from blended
ORTG/DRTG plus a home-court prior. Scores are modeled as independent Poissons
(Monte Carlo) so moneyline, spread, and total all come from one joint engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from nba_betting.features.game_features import GameFeatures

DEFAULT_N_SIMS = 20_000
# Mild rest adjustment: B2B shaves expected efficiency a bit.
B2B_POINTS_PENALTY = 1.5


@dataclass(frozen=True)
class GamePrediction:
    game_id: str
    game_date: str
    home_tricode: str
    away_tricode: str
    exp_home_score: float
    exp_away_score: float
    exp_total: float
    exp_margin: float  # home - away
    p_home_win: float
    p_away_win: float
    p_home_cover: float
    p_over: float
    fair_spread: float  # home points vs away (negative => home favored)
    fair_total: float
    market_spread: float
    market_total: float
    n_sims: int

    def to_dict(self) -> dict:
        return asdict(self)


def expected_scores(features: GameFeatures) -> tuple[float, float]:
    """Return (exp_home_pts, exp_away_pts) from pace/efficiency features."""
    pace = 0.5 * (features.home_pace + features.away_pace)
    # Blended offensive/defensive ratings → points per 100 possessions.
    home_off = 0.5 * (features.home_ortg + features.away_drtg)
    away_off = 0.5 * (features.away_ortg + features.home_drtg)

    home_pts = pace * home_off / 100.0
    away_pts = pace * away_off / 100.0
    home_pts += features.home_court_points

    if features.home_b2b:
        home_pts -= B2B_POINTS_PENALTY
    if features.away_b2b:
        away_pts -= B2B_POINTS_PENALTY

    return max(home_pts, 1.0), max(away_pts, 1.0)


def simulate_scores(
    exp_home: float,
    exp_away: float,
    *,
    n_sims: int = DEFAULT_N_SIMS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent Poisson draws for home/away scores."""
    rng = rng or np.random.default_rng(0)
    home = rng.poisson(lam=exp_home, size=int(n_sims)).astype(float)
    away = rng.poisson(lam=exp_away, size=int(n_sims)).astype(float)
    return home, away


def derive_market_probs(
    home_scores: np.ndarray,
    away_scores: np.ndarray,
    *,
    market_spread: float = 0.0,
    market_total: float | None = None,
) -> dict[str, float]:
    """Derive ML / spread / total probabilities from simulated scores.

    ``market_spread`` is the home spread (e.g. -3.5 means home -3.5).
    Home covers when ``home - away + market_spread > 0``.
    """
    margin = home_scores - away_scores
    total = home_scores + away_scores
    fair_total = float(total.mean()) if market_total is None else float(market_total)

    # Moneyline: ignore pushes (OT not modeled); break ties 50/50.
    home_wins = margin > 0
    away_wins = margin < 0
    ties = margin == 0
    p_home = float(home_wins.mean() + 0.5 * ties.mean())
    p_away = float(away_wins.mean() + 0.5 * ties.mean())

    cover = (margin + market_spread) > 0
    cover_push = (margin + market_spread) == 0
    p_cover = float(cover.mean() + 0.5 * cover_push.mean())

    line_total = fair_total if market_total is None else float(market_total)
    overs = total > line_total
    total_push = total == line_total
    p_over = float(overs.mean() + 0.5 * total_push.mean())

    # Fair spread: median margin, negated so negative => home favorite (sportsbook style).
    fair_spread = -float(np.median(margin))

    return {
        "p_home_win": p_home,
        "p_away_win": p_away,
        "p_home_cover": p_cover,
        "p_over": p_over,
        "fair_spread": fair_spread,
        "fair_total": float(total.mean()),
        "exp_margin": float(margin.mean()),
        "line_total_used": line_total,
    }


def predict_game(
    features: GameFeatures,
    *,
    market_spread: float = 0.0,
    market_total: float | None = None,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = 0,
) -> GamePrediction:
    """Full pipeline: features → expected scores → sims → market probabilities."""
    exp_home, exp_away = expected_scores(features)
    rng = np.random.default_rng(seed)
    home_s, away_s = simulate_scores(exp_home, exp_away, n_sims=n_sims, rng=rng)
    stats = derive_market_probs(
        home_s,
        away_s,
        market_spread=market_spread,
        market_total=market_total if market_total is not None else exp_home + exp_away,
    )
    return GamePrediction(
        game_id=features.game_id,
        game_date=features.game_date,
        home_tricode=features.home_tricode,
        away_tricode=features.away_tricode,
        exp_home_score=float(exp_home),
        exp_away_score=float(exp_away),
        exp_total=float(exp_home + exp_away),
        exp_margin=float(exp_home - exp_away),
        p_home_win=stats["p_home_win"],
        p_away_win=stats["p_away_win"],
        p_home_cover=stats["p_home_cover"],
        p_over=stats["p_over"],
        fair_spread=stats["fair_spread"],
        fair_total=stats["fair_total"],
        market_spread=float(market_spread),
        market_total=float(
            market_total if market_total is not None else exp_home + exp_away
        ),
        n_sims=int(n_sims),
    )
