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

Skeleton + schedule/box ingest + **as-of team features** + **simple game score model**
(ML / spread / total from one engine). Odds adapter and props are not in yet.

## Layout

```
src/nba_betting/
  data/       # schedule, box scores (odds adapter later)
  features/   # as-of pace / ORTG / DRTG / rest / B2B
  models/     # Poisson score engine → ML / spread / total
  pricing/    # de-vig, fair odds, edge, Kelly
  backtest/   # walk-forward + metrics
  slate/      # score a date from cache; lock/grade later
data/
  raw/
    schedule/{season}.parquet   # also .csv mirror
    boxscores/{season}_team.parquet
  processed/
  blind_predictions.jsonl
tests/
  fixtures/   # offline CSV fixtures (ingest + model)
```

Raw dumps under `data/raw/` are **gitignored**. Override the root with
`--data-dir` or `NBA_BETTING_DATA_DIR`.

## Requirements

- Python **3.11+**
- Optional: [uv](https://github.com/astral-sh/uv) for installs

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy env template (no real secrets in git):

```bash
cp .env.example .env
```

## Pull data (`nba_api`)

```bash
nba-betting data pull-schedule --season 2024-25
nba-betting data pull-boxes --season 2024-25
nba-betting data pull --season 2024-25          # both
nba-betting data pull --season 2024-25 --force  # re-fetch
```

Caches land at `data/raw/schedule/` and `data/raw/boxscores/` (parquet + csv).
After the first pull, loads are offline.

## Features + game model

As-of ratings use **only** team games with `game_date < slate date` (no same-day /
future leakage):

- Possessions ≈ `FGA - OREB + TOV + 0.44 * FTA`
- Pace (per 48), ORTG / DRTG from trailing window (default 20)
- Rest days + B2B flags; home-court prior (~2.5 pts)

Expected scores blend pace with offense/defense ratings; independent Poisson Monte
Carlo yields moneyline, home-cover (vs `--spread`), and over (vs `--total`) from the
**same** `(home_score, away_score)` draws.

### Score a date (offline from cache)

```bash
nba-betting model score --season 2024-25 --date 2024-10-25
nba-betting slate score --season 2024-25 --date 2024-10-25 \
  --spread -3.5 --total 220 --out data/processed/slate_2024-10-25.json
```

## Tests

Default suite is offline:

```bash
pytest
```

Optional live ingest smoke:

```bash
RUN_LIVE_NBA_API=1 pytest -m live
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `ODDS_API_KEY` | The Odds API key (empty until configured) |
| `ODDS_REGIONS` | Odds regions (default `us`) |
| `ODDS_MARKETS` | Featured markets: `h2h,spreads,totals` |
| `NBA_BETTING_DATA_DIR` | Optional override for the data root |

## Out of scope (so far)

Odds API adapter, edge/Kelly pricing, props, scrapers, Next.js / frontend, secrets.
