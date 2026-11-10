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

Package skeleton + **schedule / box-score ingest** (`nba_api`). No game model, odds
adapter, or props yet.

## Layout

```
src/nba_betting/
  data/       # schedule, box scores (odds adapter later)
  features/   # as-of team/player features
  models/     # game score engine (+ props later)
  pricing/    # de-vig, fair odds, edge, Kelly
  backtest/   # walk-forward + metrics
  slate/      # daily CLI: build board, lock, grade
data/
  raw/
    schedule/{season}.parquet   # also .csv mirror
    boxscores/{season}_team.parquet
  processed/
  blind_predictions.jsonl
tests/
  fixtures/data/   # offline CSV fixtures for unit tests
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

## Pull data (`nba_api`)

Season strings use stats.nba.com form, e.g. `2024-25`.

```bash
# Schedule (ScheduleLeagueV2) → data/raw/schedule/2024-25.parquet
nba-betting data pull-schedule --season 2024-25

# Team game logs / box lines (LeagueGameLog) → data/raw/boxscores/2024-25_team.parquet
nba-betting data pull-boxes --season 2024-25

# Both
nba-betting data pull --season 2024-25

# Re-fetch
nba-betting data pull --season 2024-25 --force

# Optional player logs
nba-betting data pull-boxes --season 2024-25 --level player
```

After the first successful pull, `load_schedule` / `load_boxscores` (and a second CLI
pull without `--force`) read only the local cache — no network.

## Other commands

```bash
nba-betting --help
nba-betting version
nba-betting slate --help
python -m nba_betting --help
```

## Tests

Default suite is offline (fixtures + injectable fetchers):

```bash
pytest
```

Optional live smoke (hits stats.nba.com):

```bash
RUN_LIVE_NBA_API=1 pytest -m live
```

## Environment variables

See `.env.example`:

| Variable | Purpose |
|----------|---------|
| `ODDS_API_KEY` | The Odds API key (empty until configured) |
| `ODDS_REGIONS` | Odds regions (default `us`) |
| `ODDS_MARKETS` | Featured markets: `h2h,spreads,totals` |
| `NBA_BETTING_DATA_DIR` | Optional override for the data root |

## Out of scope (so far)

Game model, Odds API adapter, props, scrapers, Next.js / frontend, committed secrets.
