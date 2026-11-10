# nbaML / nba-betting

Python CLI that prices NBA game markets (moneyline, spread, totals) and v1.5 player
props, finds edge vs books, and tracks walk-forward results. No web UI in v1.

## Status

Ingest + features/model + odds/edges + backtest/ledger + **player props (v1.5)**.
No DFS/PrizePicks, no live betting.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Pull team + player caches (player logs feed prop minutes priors):

```bash
nba-betting data pull --season 2024-25
nba-betting data pull-boxes --season 2024-25 --level player
```

## Game slate

```bash
nba-betting model score --season 2024-25 --date 2024-10-25
nba-betting odds pull
nba-betting slate edges --season 2024-25 --date 2024-10-25 --min-edge 0.03
nba-betting backtest run --season 2024-25
nba-betting ledger lock --season 2024-25 --date 2024-10-25
nba-betting ledger grade --season 2024-25
```

## Player props (v1.5)

As-of **minutes priors** from cached player game logs (`game_date < slate date`):
per-minute PTS/REB/AST/3PM rates × expected minutes → projections; PRA = P+R+A.
Over/under probabilities use a Poisson model on the projected mean.

```bash
# Model board (offline from player box cache)
nba-betting props score --season 2024-25 --date 2024-10-25

# Prop odds (per-event Odds API; paid tier may be required) — degrades without key
nba-betting odds pull-props

# Edges vs book lines
nba-betting props edges --season 2024-25 --date 2024-10-25 --min-edge 0.03
```

### Odds API prop market keys

| Key | Stat |
|-----|------|
| `player_points` | points |
| `player_rebounds` | rebounds |
| `player_assists` | assists |
| `player_threes` | made threes |
| `player_points_rebounds_assists` | PRA |

Endpoint shape:

```text
GET /v4/sports/basketball_nba/events/{eventId}/odds
  ?regions=us&markets=player_points,...&oddsFormat=american&apiKey=…
```

Without `ODDS_API_KEY`, `odds pull-props` / `props edges` print a clear message and
do not crash (model-only projections still work via `props score`).

## Tests

```bash
pytest
```

## Environment

See `.env.example` (`ODDS_API_KEY`, `ODDS_PROP_MARKETS`, …). Never commit real keys.

## Out of scope

DFS / PrizePicks, live betting, alternate prop ladders, scrapers, frontend.
