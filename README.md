# ML Day Trader

A machine-learning day trader for US stocks. While the market is open it pulls fresh
5-minute bars for a universe of liquid stocks, predicts each stock's move over the next
hour, and prints ranked **buy** and **sell/short** recommendations. In autonomous mode it
also trades those recommendations in a **simulated account with fake money**, so you can
watch its day-to-day performance build up.

By default it never places a real order: trades go to a built-in simulator and the only
network traffic is downloading market data. Optionally it can send its orders to **your own
Alpaca account** - Alpaca's paper service, or a funded account - see
[Connecting an Alpaca account](#connecting-an-alpaca-account). Nothing here is investment advice.

## Quick start

```bash
cd ~/Documents/"ML day trader"
bash setup.sh                    # one time: private Python environment + libraries (~2 min)

./trader demo                    # offline, ~1 min: watch the whole thing run on a synthetic market
./trader backtest                # download ~2 months of real bars, train, walk-forward backtest
./trader replay --days 3         # watch it trade the last 3 real sessions it had never seen
./trader run --capital 100000    # live: recommendations + autonomous paper trading
```

`run` keeps going until you press Ctrl+C. Outside market hours (9:30-16:00 Eastern,
Mon-Fri) it waits for the next open. A laptop that goes to sleep stops the loop, so on a
Mac start it with `caffeinate -i ./trader run --capital 100000`.

While it runs, open `reports/dashboard.html` in a browser: equity curve, profit and loss
by day, open positions, the latest recommendations and every trade. It refreshes itself.

## Commands

| Command | What it does |
|---|---|
| `./trader run --capital N` | Live loop with an autonomous paper account that starts with N dollars |
| `./trader run --advise-only` | Live loop, recommendations only, no simulated trading |
| `./trader backtest` | Walk-forward backtest on real history; writes `reports/backtest.html` |
| `./trader replay --days N` | Re-runs the last N real sessions through the live engine, using a model trained only on earlier data (separate `replay/` account) |
| `./trader demo` | Offline demo on generated data (separate `demo/` account) |
| `./trader report` | Performance of the paper account so far; refreshes the dashboard |
| `./trader status` | Data source, market hours, model quality, account balance |
| `./trader train` | Refresh data and retrain now |
| `./trader tune` | Compare prediction horizons (30 minutes to 3 hours) after costs and say which did best |
| `./trader run --broker alpaca-paper` | Same loop, but orders go to your Alpaca paper account |
| `./trader run --broker alpaca-live` | Orders go to your funded Alpaca account - real money, off until you enable it |
| `./trader flatten --broker ...` | Close everything this program holds at Alpaca and cancel its resting orders |
| `./trader reset --capital N` | Archive the paper account and start a new one |

Useful options on any command: `--universe AAPL,MSFT,NVDA,...` (at least 8 tickers),
`--provider yahoo|alpaca`, `--config other.yaml`, `--alpaca-key KEY_ID` (see below).

## What you see while it runs

```
[10:15:06 ET] bar 10:10-10:15  72/74 symbols fresh  model through 2026-11-16 (IC +0.021, t 2.9)
  Buy candidates  (next 60 min, relative to the market)
  action      symbol     price  expected    cost  net edge    hit      stop
  BUY         NVDA      142.31    +11.4bp   3.0bp    +8.4bp  58.1%    140.90
  watch       AMD       163.20     +3.9bp   3.0bp    +0.9bp  53.1%    161.75
  Sell / short candidates  ...
  >> BOUGHT 105 NVDA @ 142.33  expected +11.4bp, net edge +8.4bp, stop 140.90
  << SOLD 220 UBER @ 71.02  (signal)  P&L +$34.74
  Account: equity $100,308.79  today +$308.79 (+0.31%)  cash $90,302.88  9 open  gross 100%  market exposure -21%
```

- **expected** - the model's predicted move over the next hour, in basis points (1 bp = 0.01%),
  relative to the rest of the universe.
- **cost** - the simulated round-trip cost of trading it (spread + slippage, both ways).
- **net edge** - expected minus cost. Only rows with enough net edge are marked BUY or
  SELL/SHORT; the rest are "watch".
- **hit** - how often signals of this strength pointed the right way in validation.
- **stop** - where the paper account would cut the position.

(The numbers above are an illustration of the layout, not real results.)

## How it decides

Every choice below is aimed at the same thing: only take trades with positive expected
value after costs, and never believe a number that was not measured out-of-sample.

1. **Data.** 5-minute bars for ~70 liquid stocks and ETFs (Yahoo) or ~215 (Alpaca),
   cached locally so the history keeps growing beyond what the source serves in one go.
2. **Features (77).** Momentum and reversal over 5 minutes to 10 days, distance from VWAP,
   the day's and prior day's highs and lows, opening-range position, relative volume
   ("is this stock in play today?"), signed volume flow, volatility regime, market beta
   and market-relative strength, bar shapes, time of day, and what the stock did at this
   time of day on recent days. All measured in units of the stock's own expected
   volatility so one model fits every stock. A test proves no feature can see the future.
3. **Model.** Gradient-boosted trees blended with a heavily regularised linear model.
   It predicts the next 60 minutes' return relative to the universe.
4. **Validation.** Strictly walk-forward: train on the past, test on the days that follow,
   repeat. Model size, blend weights and the "how much to trust it" calibration are all
   chosen on those unseen days. Predicted edges are then shrunk three ways: by the
   measured calibration slope, by how statistically solid the evidence is, and by a flat
   25% haircut. If validation shows no real edge, predictions shrink to about zero and the
   trader simply stays out.
   **Strongest signals** get a second look: the top 20% down to the top 0.5% are each
   checked against what actually happened, session by session. If a group did reliably
   better (or worse) than the single scaling implies - by more than three standard errors,
   over at least 15 sessions - its expected edge is adjusted toward what was observed. A
   big average produced by a few lucky sessions is ignored, and a signal's direction is
   never reversed. The validation report shows the result for each group.
5. **Trading rule.** Enter only when the shrunk expected move beats the full round-trip
   cost. Exit when the model's view for that side fades, at a volatility-based stop, or
   at 3:55 pm - nothing is held overnight.
6. **Risk.** Each position risks 0.5% of equity to its stop and is capped at 15% of
   equity; at most 10 positions; total exposure capped at 1x equity; beta-weighted
   long-minus-short exposure capped at 30%, so results reflect stock picking rather than
   the market's direction; a 3% daily loss flattens everything until tomorrow.
7. **Fills.** Buys fill at the ask and sells at the bid when live quotes exist; otherwise
   at the last price plus an assumed half-spread, and always plus slippage.
8. **Learning.** After each close the day's bars are added and the model is retrained and
   re-validated.

The backtester and the live loop share the same decision code and the same simulated
broker; a test replays a session through both and checks they take the same trades.

## Data sources

**Yahoo Finance (default, nothing to set up).** Free and keyless, but unofficial: about
60 days of 5-minute history, one request per symbol, and occasional rate limiting. Fine
for getting started.

**Alpaca (recommended, free).** Create a free account at https://alpaca.markets, generate
API keys, copy `.env.example` to `.env` and paste them in. The trader then switches
automatically: real-time quotes with bid/ask, about six months of history to train on,
and the larger universe. The free feed is IEX, one exchange's slice of the market, which
is real-time but thinner than the full tape. These keys are used only to read market data
(unless you also choose `--broker alpaca-paper`).

Switching from Yahoo to Alpaca:

1. Give it your keys, either way:
   - **On the command line:** `./trader run --alpaca-key YOUR_KEY_ID`. It then asks for the
     secret without showing it. You can also pass `--alpaca-secret YOUR_SECRET`, but then
     the secret is saved in your shell history, so the prompt is safer.
   - **In `.env`:** copy `.env.example` to `.env` and fill in `APCA_API_KEY_ID` and
     `APCA_API_SECRET_KEY`. Then no options are needed.

   Command-line keys win over `.env` for that run and are never written to disk. Keys from
   your Alpaca **paper** account work for data and are the safer choice; command-line keys
   are also used for `--broker alpaca-paper`. Live-trading keys are only ever read from
   `.env`, never from the command line.
2. Stop the trader (Ctrl+C) and start it again with the keys.
   `./trader status --alpaca-key YOUR_KEY_ID` should say `Data provider: alpaca`.

On the first Alpaca start it downloads about six months of bars for ~215 symbols (a few
minutes) into its own cache, `data_cache/alpaca/`, separate from Yahoo's, and retrains the
model on that history. The caches are kept apart because IEX reports only a fraction of the
volume Yahoo does, so mixing them would distort the volume features. The paper account
carries on as before; trades from that day on are driven by the Alpaca-trained model. To go
back to Yahoo, set `data.provider: yahoo` in `config.yaml`.

## Connecting an Alpaca account

`--broker` chooses where the autonomous trades go. Everything else - data, model,
recommendations, risk limits - is identical in all three modes.

| `--broker` | Orders go to | Money |
|---|---|---|
| `sim` (default) | the built-in simulator | fake |
| `alpaca-paper` | Alpaca's paper-trading service | fake, but real order plumbing |
| `alpaca-live` | your funded Alpaca account | **real** |

**Paper account (do this first).**

1. In the Alpaca dashboard, switch to the paper account and generate API keys.
2. Put them in `.env` (copy `.env.example`). If your data keys are already paper keys,
   there is nothing more to add.
3. `./trader run --broker alpaca-paper --capital 25000`

`--capital` is the most the program may use; it never uses more than the account holds.

**Live account.** Three deliberate steps, so it cannot happen by accident:

1. Put the live keys in `.env` as `ALPACA_LIVE_KEY_ID` / `ALPACA_LIVE_SECRET_KEY`. Live keys
   are only ever read from those two names.
2. Set `broker: allow_live: true` in `config.yaml`.
3. `./trader run --broker alpaca-live --capital 2000` and type the confirmation phrase it
   asks for. It asks every time.

**How it behaves with a real broker**

- It keeps its own ledger (`state_alpaca_paper/`, `state_alpaca_live/`) and its own
  dashboard (`reports/alpaca_paper/`, `reports/alpaca_live/`), separate from the simulator's.
- It only trades what it opened. It refuses to start in an account that already holds
  other positions unless you set `broker.allow_shared_account: true`, and even then it
  never touches them.
- Entries are whole-share limit orders priced at most `limit_offset_bps` past the quote.
  An entry that does not fill within `fill_timeout_seconds` is cancelled, not chased.
- Exits try the same limit first, then a market order for anything left, so a position
  is not left stranded.
- Every position gets a stop order resting at Alpaca, so the stop still works if this
  program or the computer stops. The program also reconciles with Alpaca every 30 seconds:
  a stop that filled there, or a position you closed by hand, is picked up.
- No single order exceeds `broker.max_order_notional` dollars.
- Creating an empty file named `STOP` in this folder blocks all new entries (exits still work).
- Pressing Ctrl+C stops the program but does **not** close positions. Run
  `./trader flatten --broker alpaca-live` to close them.
- If short selling is not enabled on the account it runs long-only.

**Before using real money**

- The Alpaca order code has been tested against an imitation of Alpaca's API, not against
  Alpaca itself. Run `alpaca-paper` for several sessions and compare the program's ledger
  with the Alpaca dashboard before considering `alpaca-live`.
- Run it on a machine that stays awake and online through the close. If it is not
  running at 3:55 pm, positions stay open overnight and their stop orders expire.
- Your broker's rules apply and are not modelled: day-trading limits for small margin
  accounts, short-sale availability, buying power. An order Alpaca rejects is reported
  and skipped.
- A model that shows no edge in validation will mostly sit in cash; one that shows an
  edge on paper can still lose real money.

## Files it writes

| Path | Contents |
|---|---|
| `state/daily.csv` | One row per session: P&L, return, trades, winners |
| `state/trades.csv` | Every closed trade with the model's prediction next to the result |
| `state/equity.csv` | Account value at every bar |
| `state/signals.csv` | Every recommendation it printed |
| `state/portfolio.json` | The account itself (it resumes from here after a restart) |
| `reports/dashboard.html` | The live dashboard |
| `models/model.json` | The latest validation report in plain text |
| `data_cache/` | Downloaded bars |

## Settings

Everything is in `config.yaml`, with comments. The ones most worth knowing:

- `strategy.capital` - starting fake money (or `--capital` when you first run).
- `strategy.allow_short: false` - long-only.
- `strategy.max_gross_exposure` - 1.0 means no leverage; 2.0 doubles both gains and losses.
- `strategy.min_edge_bps` - raise it for fewer, pickier trades.
- `model.tail_calibration: false` - switch the strongest-signal calibration off.
- `costs.*` - how pessimistic simulated fills are. Lowering these makes results look
  better without making them more real.
- `model.horizon_bars` - what is predicted (12 bars = 60 minutes). `./trader tune` compares
  the alternatives on your data; trust it more as the history grows.
- `data.universe` - `core`, `liquid`, your own list, or a text file of tickers.

## What to expect

- **Finding a real edge in liquid stocks with public bar data is hard.** Predictable
  moves over an hour are a few basis points; trading costs are a few basis points. On many
  days the honest answer is "nothing clears the hurdle" and the trader will sit in cash.
  That is it working as designed, not a fault.
- **The validation report is the thing to read first.** `./trader status` shows the
  out-of-sample rank IC and its t-statistic, and a table of how the strongest signals did
  after costs. Read the "sessions up" column next to any impressive "net" figure: a group
  that made money in 12 of 22 sessions has not shown an edge, whatever its average. A t-statistic under about 2.5 means no dependable edge has been shown yet.
- **Short backtests prove little.** With Yahoo's ~60 days of history the walk-forward
  backtest covers only a few weeks. The paper account's forward record, built one real
  day at a time, is the trustworthy test - give it a few weeks before judging.
- **Paper results flatter reality somewhat.** No simulated fill moves the market, shorts
  are always available, and account rules for day trading on margin are not modelled.
- **The demo is not evidence.** Its market has a pattern planted in it on purpose so
  there is something to find.

## Troubleshooting

- *"No market data could be downloaded"* - check the connection; Yahoo may be rate
  limiting, so wait a few minutes or add Alpaca keys.
- *"Only N usable sessions of history"* - the model needs about 27 trading days of bars.
  Yahoo normally provides enough on the first download.
- *Symbols skipped with "no data"* - a ticker was renamed or delisted; remove it or ignore it.
- *Start over* - `./trader reset --capital 100000` (the old account is archived, not deleted).

## Tests

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests -q
```

76 tests: the market calendar against the NYSE schedule, data parsing, a no-look-ahead
proof for the features, broker accounting, a check that a pure random-walk market
produces no edge and no trades, a check that a planted pattern is found and traded
profitably, full simulated sessions through the live engine, and the Alpaca order handling against an
imitation of Alpaca's API.
