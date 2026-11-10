# nbaML / nba-betting

Python CLI that prices NBA game markets (moneyline, spread, totals), compares model
odds to books, and tracks walk-forward results. No web UI in v1 — JSON artifacts and
a daily slate CLI only.

## Goal

- Ingest schedule / box scores (primary: `nba_api`) and odds (The Odds API)
- Build as-of team pace/efficiency features → game score distribution
- Derive fair ML / spread / total prices and flag edges
- Walk-forward backtest + blind lock/grade ledger

Props (points / rebounds / assists / threes / PRA) are **v1.5**.

## Status

Ingest + features/model + odds/edges + **walk-forward backtest + blind ledger**
(Phase A results-only grading + paper ¼ Kelly). Player props are not in yet.

## Layout

```
src/nba_betting/
  data/       # schedule, boxes, odds adapter
  features/   # as-of pace / ORTG / DRTG / rest / B2B
  models/     # Poisson score engine → ML / spread / total
  pricing/    # de-vig, fair American, edge, ¼ Kelly
  backtest/   # walk-forward + metrics + paper bankroll
  slate/      # score, edges, blind lock/grade
data/
  raw/…
  processed/
  blind_predictions.jsonl
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Daily loop (high level)

```bash
nba-betting data pull --season 2024-25
nba-betting odds pull                          # needs ODDS_API_KEY
nba-betting slate score --season 2024-25 --date 2024-10-25
nba-betting slate edges --season 2024-25 --date 2024-10-25 --min-edge 0.03
nba-betting ledger lock --season 2024-25 --date 2024-10-25
# after finals land in the schedule cache:
nba-betting ledger grade --season 2024-25
```

## Walk-forward backtest

For each game date \(d\), ratings/features use **only** games with `date < d`.
Phase A grades model ML favorites and model fair spread/total vs actual scores
(no historical odds required). Paper bankroll applies flat $1 and **¼ Kelly**
(cap **5%** bankroll) on ML picks at assumed **-110** juice (labeled as such).

```bash
nba-betting backtest run --season 2024-25
nba-betting backtest run --season 2024-25 --start 2024-10-24 --out data/processed/bt.csv
```

Metrics include ML hit rate, Brier, log loss, spread ATS / totals hit rates, and
paper flat/Kelly ROI.

## Blind ledger

Locks append to `data/blind_predictions.jsonl`. Re-locking the same `game_id` is a
no-op. `ledger grade` fills finals from the cached schedule and **never** rewrites
locked prediction fields.

```bash
nba-betting ledger lock --season 2024-25 --date 2024-10-25
nba-betting ledger grade --season 2024-25
```

## Odds + edges

```bash
nba-betting odds pull
nba-betting slate edges --season 2024-25 --date 2024-10-25
```

Without `ODDS_API_KEY`, `odds pull` prints a clear message and exits 0.

## Tests

```bash
pytest
```

Offline fixtures only by default (no real Odds API key). Optional live ingest:

```bash
RUN_LIVE_NBA_API=1 pytest -m live
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `ODDS_API_KEY` | The Odds API key (empty until configured) |
| `ODDS_REGIONS` | default `us` |
| `ODDS_MARKETS` | default `h2h,spreads,totals` |
| `NBA_BETTING_DATA_DIR` | optional data root override |

## Out of scope (so far)

Player props (v1.5), historical odds CLV pipeline, scrapers, frontend, secrets in git.
