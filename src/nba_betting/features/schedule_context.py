"""Schedule-density features from prior tip dates only."""

from __future__ import annotations

import pandas as pd


def schedule_density_asof(
    prepared_team_dates: pd.Series,
    asof_date: str | pd.Timestamp,
    *,
    lookback_days: int = 7,
) -> dict[str, float]:
    """Count prior games in a lookback window and 3-in-4 flag.

    ``prepared_team_dates`` is a Series of prior ``game_date`` values for one team
    (caller must already filter ``game_date < asof``).
    """
    asof = pd.Timestamp(asof_date).normalize()
    if prepared_team_dates is None or len(prepared_team_dates) == 0:
        return {
            "games_last_n": 0.0,
            "three_in_four": 0.0,
            "lookback_days": float(lookback_days),
        }
    dates = pd.to_datetime(prepared_team_dates).dt.normalize().sort_values()
    dates = dates.loc[dates < asof]
    window_start = asof - pd.Timedelta(days=int(lookback_days))
    in_window = dates.loc[dates >= window_start]
    # 3-in-4: >=3 games with tip in [asof-3d, asof) i.e. playing 3rd in 4 nights
    four_start = asof - pd.Timedelta(days=3)
    in_four = dates.loc[dates >= four_start]
    return {
        "games_last_n": float(len(in_window)),
        "three_in_four": 1.0 if len(in_four) >= 2 else 0.0,  # already 2 prior in 3 days ⇒ tonight is 3rd in 4
        "lookback_days": float(lookback_days),
    }
