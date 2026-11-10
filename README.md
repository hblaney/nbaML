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

Skeleton + ingest + as-of features/model + **Odds API adapter + edge table**.
Kelly staking, full backtest harness, and props are not in yet.

## Layout

```
src/nba_betting/
  data/       # schedule, boxes, odds adapter
  features/   # as-of pace / ORTG / DRTG / rest / B2B
  models/     # Poisson score engine → ML / spread / total
  pricing/    # de-vig, fair American, edge join
  backtest/   # walk-forward + metrics (later)
  slate/      # score + edges from cache
data/
  raw/
    schedule/{season}.parquet
    boxscores/{season}_team.parquet
    odds/nba_*.json
  processed/
  blind_predictions.jsonl
```

Raw dumps under `data/raw/` are **gitignored**.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# later: set ODDS_API_KEY in .env (never commit it)
```

## Pull schedule / boxes

```bash
nba-betting data pull --season 2024-25
```

## Features + model score

```bash
nba-betting model score --season 2024-25 --date 2024-10-25
nba-betting slate score --season 2024-25 --date 2024-10-25
```

As-of ratings use only games with `game_date < slate date`. One Poisson engine
drives moneyline, spread, and total probabilities.

## Odds + edges

Featured NBA markets via The Odds API (`basketball_nba`: `h2h,spreads,totals`,
`regions=us`, American odds).

```bash
# Requires ODDS_API_KEY; prints a clear message and exits 0 if missing
nba-betting odds pull

# Join model slate to latest odds cache (or --odds-file / --live)
nba-betting slate edges --season 2024-25 --date 2024-10-25
nba-betting slate edges --season 2024-25 --date 2024-10-25 --min-edge 0.03 --out edges.json
```

Edge = model probability − de-vigged book implied. Without a key or cache, `slate
edges` degrades to a clear no-odds message (no crash).

## Tests

```bash
pytest
```

Odds tests mock HTTP — **no real API key required**. Optional live `nba_api` smoke:

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

Kelly / bankroll, walk-forward backtest harness, props, scrapers, frontend, secrets in git.
