"""Sharp-weighted multi-book consensus pricing and stale-outlier detection.

Combines odds across sportsbooks using:
1. Canonical bookmaker sharpness tiers (Pinnacle / LowVig / BetOnline / Betfair
   weighted above mid-tier US books DraftKings / FanDuel / Caesars, which in
   turn outweigh soft high-hold recreational books).
2. Inverse-overround weighting so tighter-hold quotes receive proportionally
   more influence even within the same tier.
3. Favorite-longshot-bias-aware devigging (Shin, Power, or Probit by default).
4. Point-adjusted Gaussian margin/total translation when books hang different
   spreads or totals (e.g. ``-6`` vs ``-6.5`` or ``221.5`` vs ``222.5``).
5. Leave-one-out (LOO) sharp consensus for stale-outlier detection so a rogue
   book quote never contaminates the benchmark used to score it.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.stats import norm

from nba_betting.pricing.american import american_to_implied, implied_to_american
from nba_betting.pricing.devig import devig_probabilities, normalize_devig_method
from nba_betting.pricing.kelly import american_to_decimal

DEFAULT_MARGIN_SIGMA = 11.8
DEFAULT_TOTAL_SIGMA = 17.8

# Sharpness tier priors aligned with nba-frontier-signals-spec.md §2.1:
# Pinnacle (10.0), LowVig.ag (5.0), BetOnline.ag (4.0) anchor consensus;
# mid-tier liquid US books (FanDuel 1.5, DraftKings 1.2, Caesars 0.75) > soft retail (0.5).
BOOK_SHARPNESS_WEIGHTS: dict[str, float] = {
    "pinnacle": 10.0,
    "circa": 8.0,
    "bookmaker": 8.0,
    "cris": 8.0,
    "lowvig": 5.0,
    "betonlineag": 4.0,
    "betfair": 3.5,
    "westgate": 2.0,
    "fanduel": 1.5,
    "draftkings": 1.2,
    "caesars": 0.75,
    "espnbet": 0.5,
    "betmgm": 0.5,
    "betrivers": 0.5,
    "sugarhouse": 0.5,
    "pointsbet": 0.5,
    "unibet": 0.5,
    "bovada": 0.5,
    "betus": 0.5,
    "hollandcasino": 0.5,
    "titanbets": 0.5,
    "mybookieag": 0.5,
}

_BOOK_ALIASES: dict[str, str] = {
    "pinnacle": "pinnacle",
    "pinny": "pinnacle",
    "circa": "circa",
    "circasports": "circa",
    "bookmaker": "bookmaker",
    "bookmaker.eu": "bookmaker",
    "lowvig": "lowvig",
    "lowvig.ag": "lowvig",
    "lowvigag": "lowvig",
    "betonline": "betonlineag",
    "betonline.ag": "betonlineag",
    "betonlineag": "betonlineag",
    "betfair": "betfair",
    "betfairsportsbook": "betfair",
    "betfair_ex_eu": "betfair",
    "betfair_ex_uk": "betfair",
    "westgate": "westgate",
    "superbook": "westgate",
    "draftkings": "draftkings",
    "draft kings": "draftkings",
    "dk": "draftkings",
    "fanduel": "fanduel",
    "fd": "fanduel",
    "betmgm": "betmgm",
    "mgm": "betmgm",
    "caesars": "caesars",
    "williamhill_us": "caesars",
    "william hill": "caesars",
    "caesars sportsbook": "caesars",
    "caesars sportsbook (colorado)": "caesars",
    "caesars sportsbook (new jersey)": "caesars",
    "caesars sportsbook (tennessee)": "caesars",
    "espnbet": "espnbet",
    "espn bet": "espnbet",
    "barstool": "espnbet",
    "betrivers": "betrivers",
    "bet rivers": "betrivers",
    "sugarhouse": "sugarhouse",
    "pointsbet": "pointsbet",
    "pointsbetus": "pointsbet",
    "fanatics": "pointsbet",
    "unibet": "unibet",
    "unibet_us": "unibet",
    "bovada": "bovada",
    "betus": "betus",
    "mybookie": "mybookieag",
    "mybookie.ag": "mybookieag",
    "mybookieag": "mybookieag",
    "holland casino": "hollandcasino",
    "hollandcasino": "hollandcasino",
    "titanbets": "titanbets",
}

NON_BOOK_PROVIDERS: frozenset[str] = frozenset(
    {"consensus", "teamrankings", "accuscore", "numberfire", "bpi"}
)
LIVE_PROVIDER_IDS: frozenset[str] = frozenset({"46", "59"})


def normalize_book_key(book: str) -> str:
    """Normalize display / provider / Odds API bookmaker names to a canonical key."""
    raw = str(book or "").strip().lower()
    if raw in _BOOK_ALIASES:
        return _BOOK_ALIASES[raw]
    # Handle state-qualified names like "Caesars Sportsbook (Virginia)"
    stripped = re.sub(r"\s*\([^)]*\)\s*", "", raw).strip()
    if stripped in _BOOK_ALIASES:
        return _BOOK_ALIASES[stripped]
    alnum = re.sub(r"[^a-z0-9]+", "", stripped)
    return _BOOK_ALIASES.get(alnum, alnum or raw)


def is_valid_pregame_book(
    provider_name: str | None,
    provider_id: str | int | None = None,
) -> bool:
    """Return True iff provider represents a real pregame sportsbook (not live/synthetic)."""
    name = str(provider_name or "").strip().lower()
    pid = str(provider_id or "").strip()
    if not name:
        return False
    if "live" in name or pid in LIVE_PROVIDER_IDS:
        return False
    canon = normalize_book_key(name)
    if canon in NON_BOOK_PROVIDERS or name in NON_BOOK_PROVIDERS:
        return False
    return True


def book_sharpness_weight(book: str, *, default: float = 1.0) -> float:
    """Lookup sharpness prior weight for ``book``."""
    return float(BOOK_SHARPNESS_WEIGHTS.get(normalize_book_key(book), default))


@dataclass(frozen=True)
class TwoWayQuote:
    """Two-way sportsbook quote for moneyline, spread, or total."""

    book: str
    side_a_price: float  # home ML, home spread price, or over price
    side_b_price: float  # away ML, away spread price, or under price
    line_a: float | None = None  # home spread (e.g. -6.5) or total points (e.g. 222.5)
    market: str = "moneyline"  # "moneyline" | "spread" | "total"

    @property
    def canonical_book(self) -> str:
        return normalize_book_key(self.book)

    @property
    def overround(self) -> float:
        return american_to_implied(self.side_a_price) + american_to_implied(self.side_b_price)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConsensusResult:
    """Sharp-weighted consensus fair probabilities and market-dispersion stats."""

    market: str
    devig_method: str
    n_books: int
    prob_a: float
    prob_b: float
    fair_american_a: float
    fair_american_b: float
    consensus_line: float | None
    implied_mean: float | None
    best_price_a: float
    best_book_a: str
    best_line_a: float | None
    best_price_b: float
    best_book_b: str
    best_line_b: float | None
    synthetic_overround: float
    is_arb: bool
    mean_overround: float
    min_overround: float
    max_overround: float
    dispersion_std_a: float
    book_breakdown: list[dict[str, Any]]

    def prob_side_a_at_line(
        self,
        line_a: float | None,
        *,
        margin_sigma: float = DEFAULT_MARGIN_SIGMA,
        total_sigma: float = DEFAULT_TOTAL_SIGMA,
    ) -> float:
        """Evaluate sharp-consensus probability of side A at a specific ``line_a``."""
        if line_a is None or self.implied_mean is None or self.market == "moneyline":
            return float(self.prob_a)
        if self.market == "spread":
            z = (float(self.implied_mean) + float(line_a)) / float(margin_sigma)
            return float(np.clip(norm.cdf(z), 1e-6, 1.0 - 1e-6))
        if self.market == "total":
            z = (float(self.implied_mean) - float(line_a)) / float(total_sigma)
            return float(np.clip(norm.cdf(z), 1e-6, 1.0 - 1e-6))
        return float(self.prob_a)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StaleOutlier:
    """Detected stale / off-market sportsbook quote vs sharp consensus."""

    book: str
    canonical_book: str
    market: str
    side: str  # "a" or "b" ("home"/"away" or "over"/"under" in higher-level wrappers)
    book_american: float
    book_decimal: float
    book_line: float | None
    consensus_line: float | None
    line_out_pts: float | None
    book_implied_vigged: float
    book_prob_devig: float
    consensus_prob: float
    consensus_fair_american: float
    ev_per_unit: float
    vig_free_edge: float
    prob_edge_vs_vigged: float
    z_score: float
    n_peer_books: int
    peer_sharp_weight_sum: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deduplicate_quotes(quotes: Sequence[TwoWayQuote]) -> list[TwoWayQuote]:
    """Keep the lowest-overround quote per canonical book (prevents state-clone overweighting)."""
    by_canon: dict[str, TwoWayQuote] = {}
    for q in quotes:
        ck = q.canonical_book
        prev = by_canon.get(ck)
        if prev is None or q.overround < prev.overround:
            by_canon[ck] = q
    return list(by_canon.values())


def _effective_book_weight(
    q: TwoWayQuote,
    *,
    use_inverse_overround: bool = True,
) -> float:
    w = book_sharpness_weight(q.book)
    if use_inverse_overround:
        hold = max(q.overround - 1.0, 0.0075)
        w = w * (0.045 / hold)
    return float(max(w, 1e-6))


def _best_quote_side(
    quotes: Sequence[TwoWayQuote],
    *,
    side: str,
    consensus_mean: float | None,
    market: str,
    margin_sigma: float,
    total_sigma: float,
    prob_a_default: float,
) -> tuple[float, str, float | None]:
    """Pick the quote offering highest expected return (or highest American when lines match)."""
    best_ev = float("-inf")
    best_price = float("-inf")
    best_book = ""
    best_line: float | None = None
    for q in quotes:
        price = float(q.side_a_price if side == "a" else q.side_b_price)
        dec = american_to_decimal(price)
        if market == "spread" and consensus_mean is not None and q.line_a is not None:
            pa = float(
                np.clip(
                    norm.cdf((consensus_mean + float(q.line_a)) / margin_sigma), 1e-6, 1.0 - 1e-6
                )
            )
        elif market == "total" and consensus_mean is not None and q.line_a is not None:
            pa = float(
                np.clip(
                    norm.cdf((consensus_mean - float(q.line_a)) / total_sigma), 1e-6, 1.0 - 1e-6
                )
            )
        else:
            pa = prob_a_default
        p = pa if side == "a" else (1.0 - pa)
        ev = p * dec - 1.0
        line_val = (
            q.line_a
            if side == "a"
            else (-float(q.line_a) if (q.line_a is not None and market == "spread") else q.line_a)
        )
        if ev > best_ev + 1e-12 or (abs(ev - best_ev) <= 1e-12 and price > best_price):
            best_ev = ev
            best_price = price
            best_book = q.book
            best_line = line_val
    return best_price, best_book, best_line


def compute_sharp_consensus(
    quotes: Sequence[TwoWayQuote],
    *,
    devig_method: str = "shin",
    use_inverse_overround: bool = True,
    deduplicate_canonical_books: bool = True,
    combine_space: str = "linear",
    margin_sigma: float = DEFAULT_MARGIN_SIGMA,
    total_sigma: float = DEFAULT_TOTAL_SIGMA,
    reference_line: float | None = None,
) -> ConsensusResult:
    """Compute sharp-weighted consensus fair odds and dispersion across ``quotes``."""
    valid = [
        q
        for q in quotes
        if q.side_a_price != 0
        and q.side_b_price != 0
        and math.isfinite(q.side_a_price)
        and math.isfinite(q.side_b_price)
    ]
    if not valid:
        raise ValueError("compute_sharp_consensus requires at least one valid TwoWayQuote")

    canon_method = normalize_devig_method(devig_method)
    market = str(valid[0].market).lower()
    pool = _deduplicate_quotes(valid) if deduplicate_canonical_books else list(valid)

    weights = np.array(
        [_effective_book_weight(q, use_inverse_overround=use_inverse_overround) for q in pool],
        dtype=float,
    )
    weights = weights / weights.sum()

    devig_a_list: list[float] = []
    overrounds: list[float] = []
    latent_means: list[float] = []
    breakdown: list[dict[str, Any]] = []

    has_lines = all(q.line_a is not None for q in pool) and market in {"spread", "total"}
    ref_line: float | None = reference_line
    if has_lines and ref_line is None:
        # Weighted median / modal line across books
        lines_arr = np.array([float(q.line_a) for q in pool], dtype=float)
        order = np.argsort(lines_arr)
        cdf = np.cumsum(weights[order])
        ref_line = float(lines_arr[order][np.searchsorted(cdf, 0.5)])

    for q, w in zip(pool, weights):
        raw_a = american_to_implied(q.side_a_price)
        raw_b = american_to_implied(q.side_b_price)
        pa, pb = devig_probabilities((raw_a, raw_b), method=canon_method)
        pa_clamped = float(np.clip(pa, 1e-6, 1.0 - 1e-6))
        overrounds.append(raw_a + raw_b)

        mu_b: float | None = None
        if has_lines and q.line_a is not None:
            if market == "spread":
                mu_b = -float(q.line_a) + float(margin_sigma) * float(norm.ppf(pa_clamped))
                pa_at_ref = float(
                    np.clip(
                        norm.cdf((mu_b + float(ref_line)) / float(margin_sigma)), 1e-6, 1.0 - 1e-6
                    )
                )
            else:
                mu_b = float(q.line_a) + float(total_sigma) * float(norm.ppf(pa_clamped))
                pa_at_ref = float(
                    np.clip(
                        norm.cdf((mu_b - float(ref_line)) / float(total_sigma)), 1e-6, 1.0 - 1e-6
                    )
                )
            latent_means.append(mu_b)
            devig_a_list.append(pa_at_ref)
        else:
            devig_a_list.append(pa_clamped)

        breakdown.append(
            {
                "book": q.book,
                "canonical_book": q.canonical_book,
                "weight": float(w),
                "side_a_price": float(q.side_a_price),
                "side_b_price": float(q.side_b_price),
                "line_a": q.line_a,
                "overround": float(raw_a + raw_b),
                "prob_a_devig": float(pa),
                "prob_b_devig": float(pb),
                "implied_mean": mu_b,
            }
        )

    probs_a = np.asarray(devig_a_list, dtype=float)
    implied_mean: float | None = None
    if has_lines and latent_means:
        implied_mean = float(np.dot(weights, np.asarray(latent_means, dtype=float)))
        assert ref_line is not None
        if market == "spread":
            prob_a = float(
                np.clip(norm.cdf((implied_mean + ref_line) / margin_sigma), 1e-6, 1.0 - 1e-6)
            )
        else:
            prob_a = float(
                np.clip(norm.cdf((implied_mean - ref_line) / total_sigma), 1e-6, 1.0 - 1e-6)
            )
    elif combine_space == "logit":
        logits = np.log(probs_a / (1.0 - probs_a))
        mean_logit = float(np.dot(weights, logits))
        prob_a = float(1.0 / (1.0 + math.exp(-mean_logit)))
    else:
        prob_a = float(np.clip(np.dot(weights, probs_a), 1e-6, 1.0 - 1e-6))

    prob_b = float(1.0 - prob_a)

    # Best executable sides across ALL valid quotes (even before deduplication).
    best_price_a, best_book_a, best_line_a = _best_quote_side(
        valid,
        side="a",
        consensus_mean=implied_mean,
        market=market,
        margin_sigma=margin_sigma,
        total_sigma=total_sigma,
        prob_a_default=prob_a,
    )
    best_price_b, best_book_b, best_line_b = _best_quote_side(
        valid,
        side="b",
        consensus_mean=implied_mean,
        market=market,
        margin_sigma=margin_sigma,
        total_sigma=total_sigma,
        prob_a_default=prob_a,
    )

    # Synthetic overround on matching lines (for ML, or at ref_line for spreads/totals).
    same_line_quotes = (
        [q for q in valid if q.line_a is not None and abs(float(q.line_a) - float(ref_line)) < 1e-9]
        if (has_lines and ref_line is not None)
        else valid
    )
    if same_line_quotes:
        min_imp_a = min(american_to_implied(q.side_a_price) for q in same_line_quotes)
        min_imp_b = min(american_to_implied(q.side_b_price) for q in same_line_quotes)
        synth_overround = float(min_imp_a + min_imp_b)
    else:
        synth_overround = float(
            american_to_implied(best_price_a) + american_to_implied(best_price_b)
        )

    disp = float(np.std(probs_a, ddof=0)) if probs_a.size > 1 else 0.0

    return ConsensusResult(
        market=market,
        devig_method=canon_method,
        n_books=int(len(pool)),
        prob_a=prob_a,
        prob_b=prob_b,
        fair_american_a=float(implied_to_american(prob_a)),
        fair_american_b=float(implied_to_american(prob_b)),
        consensus_line=ref_line,
        implied_mean=implied_mean,
        best_price_a=best_price_a,
        best_book_a=best_book_a,
        best_line_a=best_line_a,
        best_price_b=best_price_b,
        best_book_b=best_book_b,
        best_line_b=best_line_b,
        synthetic_overround=synth_overround,
        is_arb=bool(synth_overround < 1.0 - 1e-9),
        mean_overround=float(np.mean(overrounds)),
        min_overround=float(np.min(overrounds)),
        max_overround=float(np.max(overrounds)),
        dispersion_std_a=disp,
        book_breakdown=breakdown,
    )


def detect_stale_outliers(
    quotes: Sequence[TwoWayQuote],
    *,
    devig_method: str = "shin",
    min_ev: float = 0.0,
    max_ev: float = 0.15,
    min_z_score: float | None = None,
    min_peer_weight: float = 0.0,
    leave_one_out: bool = True,
    use_inverse_overround: bool = True,
    margin_sigma: float = DEFAULT_MARGIN_SIGMA,
    total_sigma: float = DEFAULT_TOTAL_SIGMA,
) -> list[StaleOutlier]:
    """Identify stale or off-market book quotes against sharp peer consensus.

    - Excludes target book ``b*`` when ``leave_one_out=True`` so a stale retail
      quote never contaminates the sharp benchmark.
    - Enforces ``sum_{b != b*} w_b >= min_peer_weight`` (pass ``4.0`` on live feeds
      to require at least one sharp anchor such as ``LowVig.ag`` / ``BetOnline.ag`` /
      ``Pinnacle`` or a deep liquid book panel).
    - Suppresses corrupt/stale feed spikes above ``max_ev`` (default ``+15%``).
    """
    valid = [
        q
        for q in quotes
        if q.side_a_price != 0
        and q.side_b_price != 0
        and math.isfinite(q.side_a_price)
        and math.isfinite(q.side_b_price)
    ]
    if len(valid) < 2:
        return []

    canon_method = normalize_devig_method(devig_method)
    market = str(valid[0].market).lower()
    full_cons = compute_sharp_consensus(
        valid,
        devig_method=canon_method,
        use_inverse_overround=use_inverse_overround,
        margin_sigma=margin_sigma,
        total_sigma=total_sigma,
    )

    # Peer devigged probabilities at consensus line for z-score dispersion.
    book_probs_a_at_ref: list[float] = []
    for q in valid:
        pa, _ = devig_probabilities(
            (american_to_implied(q.side_a_price), american_to_implied(q.side_b_price)),
            method=canon_method,
        )
        if market == "spread" and q.line_a is not None and full_cons.consensus_line is not None:
            mu_q = -float(q.line_a) + margin_sigma * float(norm.ppf(np.clip(pa, 1e-6, 1 - 1e-6)))
            pa = float(norm.cdf((mu_q + float(full_cons.consensus_line)) / margin_sigma))
        elif market == "total" and q.line_a is not None and full_cons.consensus_line is not None:
            mu_q = float(q.line_a) + total_sigma * float(norm.ppf(np.clip(pa, 1e-6, 1 - 1e-6)))
            pa = float(norm.cdf((mu_q - float(full_cons.consensus_line)) / total_sigma))
        book_probs_a_at_ref.append(float(pa))

    arr_ref = np.asarray(book_probs_a_at_ref, dtype=float)
    med_ref = float(np.median(arr_ref))
    mad_ref = float(np.median(np.abs(arr_ref - med_ref))) * 1.4826
    std_ref = float(np.std(arr_ref, ddof=0))
    scale = max(mad_ref, std_ref, 0.005)

    outliers: list[StaleOutlier] = []
    for q in valid:
        peers = [p for p in valid if p.canonical_book != q.canonical_book]
        dedup_peers = _deduplicate_quotes(peers)
        peer_weight_sum = float(sum(book_sharpness_weight(p.book) for p in dedup_peers))
        if peer_weight_sum < float(min_peer_weight):
            continue
        if leave_one_out and len(peers) >= 1:
            bench = compute_sharp_consensus(
                peers,
                devig_method=canon_method,
                use_inverse_overround=use_inverse_overround,
                margin_sigma=margin_sigma,
                total_sigma=total_sigma,
            )
        else:
            bench = full_cons

        pa_cons = bench.prob_side_a_at_line(
            q.line_a, margin_sigma=margin_sigma, total_sigma=total_sigma
        )
        pb_cons = 1.0 - pa_cons
        raw_a = american_to_implied(q.side_a_price)
        raw_b = american_to_implied(q.side_b_price)
        pa_devig, pb_devig = devig_probabilities((raw_a, raw_b), method=canon_method)

        cons_line_a = bench.consensus_line
        for side_code, price, raw_imp, p_devig, p_cons, line_val, ref_ln in (
            ("a", float(q.side_a_price), raw_a, pa_devig, pa_cons, q.line_a, cons_line_a),
            (
                "b",
                float(q.side_b_price),
                raw_b,
                pb_devig,
                pb_cons,
                (-float(q.line_a) if (q.line_a is not None and market == "spread") else q.line_a),
                (
                    -float(cons_line_a)
                    if (cons_line_a is not None and market == "spread")
                    else cons_line_a
                ),
            ),
        ):
            dec = american_to_decimal(price)
            ev = p_cons * dec - 1.0
            if ev < float(min_ev) or ev > float(max_ev):
                continue
            fair_edge = p_cons - p_devig
            vig_edge = p_cons - raw_imp
            z = fair_edge / scale
            if min_z_score is not None and z < float(min_z_score):
                continue
            line_out: float | None = None
            if line_val is not None and ref_ln is not None:
                if market == "spread":
                    # Positive means bettor gets more points than consensus (favorable stale number)
                    line_out = float(line_val) - float(ref_ln)
                elif market == "total":
                    line_out = (
                        float(ref_ln) - float(line_val)
                        if side_code == "a"
                        else float(line_val) - float(ref_ln)
                    )
            outliers.append(
                StaleOutlier(
                    book=q.book,
                    canonical_book=q.canonical_book,
                    market=market,
                    side=side_code,
                    book_american=price,
                    book_decimal=dec,
                    book_line=line_val,
                    consensus_line=ref_ln,
                    line_out_pts=line_out,
                    book_implied_vigged=raw_imp,
                    book_prob_devig=p_devig,
                    consensus_prob=p_cons,
                    consensus_fair_american=float(
                        implied_to_american(min(max(p_cons, 1e-6), 1.0 - 1e-6))
                    ),
                    ev_per_unit=float(ev),
                    vig_free_edge=float(fair_edge),
                    prob_edge_vs_vigged=float(vig_edge),
                    z_score=float(z),
                    n_peer_books=int(bench.n_books),
                    peer_sharp_weight_sum=peer_weight_sum,
                )
            )

    outliers.sort(key=lambda x: (-x.ev_per_unit, -x.z_score))
    return outliers
