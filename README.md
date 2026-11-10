# nbaML / nba-betting

Python CLI that prices NBA game markets (moneyline, spread, totals), compares model
odds to books, and tracks walk-forward results. No web UI in v1 — JSON artifacts and
a daily slate CLI only.

## Goal

- Ingest schedule / box scores (primary: `nba_api`) and odds (The Odds API)
- Build as-of team pace/efficiency features → game score distribution
- Derive fair ML / spread / total prices and flag edges
- Walk-forward backtest + blind lock/grade ledger

Props (points / rebounds / assists / threes / PRA) are **v1.5**, not this skeleton.

## Status

**Skeleton only.** Package layout, install, smoke tests, and a CLI stub. No live API
calls, models, or odds pulls yet.

## Layout

```
src/nba_betting/
  data/       # schedule, box scores, odds adapter
  features/   # as-of team/player features
  models/     # game score engine (+ props later)
  pricing/    # de-vig, fair odds, edge, Kelly
  backtest/   # walk-forward + metrics
  slate/      # daily CLI: build board, lock, grade
data/
  raw/
  processed/
  blind_predictions.jsonl
tests/
```

## Requirements

- Python **3.11+**
- Optional: [uv](https://github.com/astral-sh/uv) for installs

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or with uv:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Copy env template (no real secrets in git):

```bash
cp .env.example .env
# later: set ODDS_API_KEY in .env or as a Cursor/GitHub secret
```

## Run

CLI entry point:

```bash
nba-betting --help
nba-betting version
nba-betting slate --help
```

Module form:

```bash
python -m nba_betting --help
```

## Tests

```bash
pytest
```

## Environment variables

See `.env.example`:

| Variable | Purpose |
|----------|---------|
| `ODDS_API_KEY` | The Odds API key (empty until configured) |
| `ODDS_REGIONS` | Odds regions (default `us`) |
| `ODDS_MARKETS` | Featured markets: `h2h,spreads,totals` |

## Out of scope for this PR

Models, live Odds API / `nba_api` pulls, scrapers, Next.js / frontend, committed secrets.
