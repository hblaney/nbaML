"""Fractional Kelly sizing with a per-bet bankroll cap."""

from __future__ import annotations

# Defaults until Henry overrides (see plan).
KELLY_FRACTION = 0.25
MAX_BET_FRAC = 0.05  # 5% bankroll cap
MIN_EDGE = 0.0
STARTING_BANKROLL = 100.0
DEFAULT_AMERICAN = -110.0


def american_to_decimal(price: float) -> float:
    if price == 0:
        raise ValueError("American odds cannot be 0")
    if price > 0:
        return 1.0 + price / 100.0
    return 1.0 + 100.0 / abs(price)


def kelly_fraction(
    model_prob: float,
    american_odds: float,
    *,
    fraction: float = KELLY_FRACTION,
    max_bet_frac: float = MAX_BET_FRAC,
    min_edge: float = MIN_EDGE,
) -> float:
    """Return stake as a fraction of bankroll (0 if no edge / invalid).

    Full Kelly is ``(b*p - q) / b`` with ``b = decimal - 1``. We apply
    ``fraction`` (default ¼) and cap at ``max_bet_frac`` of bankroll.
    """
    if not 0.0 < model_prob < 1.0:
        return 0.0
    decimal = american_to_decimal(american_odds)
    b = decimal - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    full = (b * model_prob - q) / b
    # Require positive Kelly edge; optional extra edge floor.
    if full <= min_edge:
        return 0.0
    stake = full * fraction
    return float(min(stake, max_bet_frac))


def stake_amount(
    bankroll: float,
    model_prob: float,
    american_odds: float,
    *,
    fraction: float = KELLY_FRACTION,
    max_bet_frac: float = MAX_BET_FRAC,
    min_edge: float = MIN_EDGE,
) -> float:
    """Dollar stake from bankroll using ¼ Kelly + cap."""
    frac = kelly_fraction(
        model_prob,
        american_odds,
        fraction=fraction,
        max_bet_frac=max_bet_frac,
        min_edge=min_edge,
    )
    return float(bankroll) * frac
