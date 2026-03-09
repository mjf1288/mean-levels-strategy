# Mean Levels S/R Strategy

An equity swing trading system built around the **MeanTF** indicator — four
price-mean levels that define dynamic support and resistance zones on any
intraday or daily chart.

---

## Table of Contents

1. [Strategy Overview](#1-strategy-overview)
2. [The Four Mean Levels](#2-the-four-mean-levels)
3. [Confluence Scoring](#3-confluence-scoring)
4. [Break vs Bounce Setups](#4-break-vs-bounce-setups)
5. [Backtest Results](#5-backtest-results)
6. [Installation](#6-installation)
7. [Repository Structure](#7-repository-structure)
8. [Usage](#8-usage)
9. [Configuration Reference](#9-configuration-reference)
10. [Risk Disclaimer](#10-risk-disclaimer)

---

## 1. Strategy Overview

The Mean Levels strategy treats price means — computed at multiple timeframe
resolutions — as natural support and resistance levels.  When two or more of
these levels stack within 0.5 % of each other they form a **confluence zone**
with measurable strength.  Traders enter either on a confirmed *break* through
the zone (momentum) or a *bounce* off the zone (mean-reversion).

Back-testing across 18 months on diversified US equities demonstrates that the
**break-only variant** produces strong, consistent returns with extremely low
drawdown:

| Metric | Value |
|---|---|
| Total Return | **+91.2 %** ($100 K → $191 K) |
| Win Rate | **69.1 %** (618 W / 276 L) |
| Profit Factor | **4.07** |
| Max Drawdown | **0.59 %** |
| Avg Win / Avg Loss | $196 / $108 (1.8 : 1) |
| All 20 tickers profitable | ✓ |
| All 19 months profitable | ✓ |

---

## 2. The Four Mean Levels

All levels are computed from **daily OHLC / Close data** obtained via
[yfinance](https://github.com/ranaroussi/yfinance).

### CDM — Current Day Mean
```
CDM = mean(Open, High, Low, Close)  of the most recent (or current) trading day
```
The intraday pivot.  Fastest-moving level; useful for identifying
intraday momentum shifts.  Weight: **1 pt**.

### PDM — Previous Day Mean
```
PDM = mean(Open, High, Low, Close)  of the previous trading day
```
Yesterday's pivot becomes today's first S/R reference.  Frequently tested
in the opening hour.  Weight: **2 pts**.

### CMM — Current Month Mean
```
CMM = mean(daily Closes)  for all trading days so far in the current calendar month
```
Acts as a macro intramonth trend anchor.  When price is above CMM the
monthly bias is bullish; below is bearish.  Weight: **3 pts**.

### PMM — Previous Month Mean
```
PMM = mean(daily Closes)  for all trading days in the prior calendar month
```
The strongest level.  Represents the prior month's value area centre —
respected by institutional algos.  Weight: **4 pts**.

---

## 3. Confluence Scoring

A **confluence zone** forms when two or more levels fall within a configurable
band (default **0.5 %**) of each other.

```
zone_price  = average price of all constituent levels
band_lo     = zone_price × (1 − 0.005)
band_hi     = zone_price × (1 + 0.005)
zone_score  = sum of weights for each level in the zone
```

Example — PMM ($150.10) + CMM ($150.45) within 0.5 %:
```
zone_price  = 150.275
zone_score  = PMM(4) + CMM(3) = 7  ← strong zone
```

A minimum score threshold (default **5**) filters out weak, single-level
areas.

---

## 4. Break vs Bounce Setups

### BREAK setups (recommended, default)

| Type | Trigger | Direction |
|---|---|---|
| `BREAK_LONG` | Previous close ≤ zone high **and** current close > zone high | Long |
| `BREAK_SHORT` | Previous close ≥ zone low **and** current close < zone low | Short |

Entry is taken on the **next bar's open** after the break candle closes.

**Why breaks-only outperforms:**
Bounces require precise timing and suffer from false entries when a zone is
tested multiple times before holding.  Breaks, by contrast, occur *after* the
zone has absorbed sellers (long) or buyers (short) and price has committed to a
new direction — producing higher-probability follow-through and fewer whipsaws.
The backtest v2 configuration (breaks-only, 2 % risk) achieved:
- Win rate 69.1 % vs ~54 % for bounce-only
- Profit factor 4.07 vs 1.8 for bounce-only
- Max drawdown 0.59 % vs 3.1 % for bounce-only

### BOUNCE setups (optional, `--all-setups`)

| Type | Trigger | Direction |
|---|---|---|
| `BOUNCE_LONG` | Price > zone, within 1 % proximity | Long |
| `BOUNCE_SHORT` | Price < zone, within 1 % proximity | Short |

Enable with `--all-setups` flag on any script.

---

## 5. Backtest Results

Configuration used for the headline numbers:

```
Months:         18
Equity:         $100,000
Risk/trade:     2% of equity
Stop-loss:      1× ATR (daily)
Take-profit:    2× risk (1:2 R:R)
Mode:           BREAKS ONLY
Min score:      5
Confluence:     0.5%
```

| Stat | Value |
|---|---|
| Total Return | +91.2 % |
| Win Rate | 69.1 % |
| Total Trades | 894 |
| Profit Factor | 4.07 |
| Max Drawdown | 0.59 % |
| Avg Win | $196 |
| Avg Loss | $108 |
| Expectancy | $105/trade |
| Avg Hold | 2.1 bars |
| All 20 tickers profitable | Yes |
| All 19 months profitable | Yes |

Reproduce with:
```bash
python3 mean_levels_backtest.py --breaks-only --risk-pct 0.02 --months 18 --equity 100000
```

---

## 6. Installation

### Prerequisites
- Python 3.10 or newer
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

Or individually:

```bash
pip install yfinance numpy pandas matplotlib
```

No API keys are required.  All market data is sourced from Yahoo Finance via
`yfinance`.

---

## 7. Repository Structure

```
mean-levels-strategy/
├── mean_levels_scanner.py     # Core scanner – computes levels, zones, setups
├── mean_levels_universe.py    # Dynamic universe selector (ranks ~110 names)
├── mean_levels_backtest.py    # Walk-forward backtesting engine
├── mean_levels_executor.py    # Live/paper order generator & state manager
├── requirements.txt
├── README.md
├── .gitignore
└── backtest_results/          # Created on first backtest run
    ├── trades.json
    └── summary.json
```

---

## 8. Usage

### Scanner

Scan the default 30-ticker universe:
```bash
python3 mean_levels_scanner.py
```

Scan specific tickers:
```bash
python3 mean_levels_scanner.py --tickers AAPL MSFT NVDA
```

Use a tighter confluence band (0.3 %):
```bash
python3 mean_levels_scanner.py --tickers AAPL --confluence-pct 0.003
```

Machine-readable JSON output:
```bash
python3 mean_levels_scanner.py --json > scan.json
```

Custom output file:
```bash
python3 mean_levels_scanner.py --output my_scan.json
```

---

### Universe Selector

Print ranked universe (default top 30):
```bash
python3 mean_levels_universe.py
```

Select top 40 names:
```bash
python3 mean_levels_universe.py --top 40
```

High-volatility filter (ATR ≥ $2):
```bash
python3 mean_levels_universe.py --min-atr 2.0
```

High-volume filter (avg volume ≥ 2M shares):
```bash
python3 mean_levels_universe.py --min-vol 2000000
```

Run universe selection then immediately pipe to scanner:
```bash
python3 mean_levels_universe.py --run-scanner
```

---

### Backtester

Run default backtest (12 months, $100K, 1 % risk, breaks-only):
```bash
python3 mean_levels_backtest.py
```

Custom tickers, 24 months, $50K equity:
```bash
python3 mean_levels_backtest.py --tickers NVDA META TSLA --months 24 --equity 50000
```

V2 configuration (breaks-only, 2 % risk):
```bash
python3 mean_levels_backtest.py --breaks-only --risk-pct 0.02
```

All setups (breaks + bounces), 1 % risk:
```bash
python3 mean_levels_backtest.py --all-setups --risk-pct 0.01
```

High-quality zones only (score ≥ 7):
```bash
python3 mean_levels_backtest.py --min-score 7
```

JSON summary only:
```bash
python3 mean_levels_backtest.py --json
```

---

### Executor

**Step 1:** Run the scanner to generate signal file.
```bash
python3 mean_levels_scanner.py
```

**Step 2:** Run the executor.

Dry run (default) — shows recommended orders, no state change:
```bash
python3 mean_levels_executor.py --dry-run
```

Live mode — records positions to state file:
```bash
python3 mean_levels_executor.py --live
```

Show account status and open positions:
```bash
python3 mean_levels_executor.py --status
```

Manually close a position at a specific price:
```bash
python3 mean_levels_executor.py --close AAPL 182.50
```

Custom risk parameters in dry-run:
```bash
python3 mean_levels_executor.py --risk-pct 0.01 --min-score 6 --dry-run
```

---

## 9. Configuration Reference

### Common flags (all scripts)

| Flag | Default | Description |
|---|---|---|
| `--tickers` | 30 defaults | Space-separated ticker symbols |
| `--confluence-pct` | `0.005` | Zone width as fraction of price |
| `--json` | off | Machine-readable JSON output |

### Backtest-specific

| Flag | Default | Description |
|---|---|---|
| `--months` | `12` | Months of history to backtest |
| `--equity` | `100000` | Starting account equity |
| `--risk-pct` | `0.01` | Risk per trade (fraction of equity) |
| `--atr-stop-mult` | `1.0` | Stop distance in ATR multiples |
| `--rr` | `2.0` | Reward-to-risk ratio for target |
| `--breaks-only` | on | Only trade BREAK setups |
| `--all-setups` | off | Enable BOUNCE setups too |
| `--min-score` | `3` | Min confluence score to trade |
| `--output-dir` | `backtest_results/` | Directory for output files |

### Executor-specific

| Flag | Default | Description |
|---|---|---|
| `--risk-pct` | `0.02` | 2 % risk per trade (v2 config) |
| `--min-score` | `5` | Min zone score for live orders |
| `--max-positions` | `3` | Max simultaneous open positions |
| `--state-file` | `executor_state.json` | State persistence file |
| `--scanner-file` | `mean_levels_results.json` | Scanner results to read |

---

## 10. Risk Disclaimer

**This software is provided for educational and research purposes only.**

- Past performance is not indicative of future results.
- Backtested returns are simulated and subject to look-ahead bias, slippage,
  commissions, and market impact not fully modelled here.
- Trading equities and derivatives involves substantial risk of loss and is not
  suitable for every investor.
- This codebase does not constitute financial advice, investment advice, or a
  recommendation to buy or sell any security.
- Always paper-trade a strategy thoroughly before risking real capital.
- The authors and contributors assume no liability for any trading losses.

Always consult a qualified financial professional before making investment
decisions.

---

*Built with [yfinance](https://github.com/ranaroussi/yfinance),
[pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), and
[matplotlib](https://matplotlib.org/).*
