# Mean Levels S/R Strategy

An equity swing trading system built around the **MeanTF** indicator — four
price-mean levels that define dynamic support and resistance zones on any
daily chart. Enhanced with **DSS Bressert + Lyapunov HP** regime filters
for directional gating.

---

## Table of Contents

1. [Strategy Overview](#1-strategy-overview)
2. [The Four Mean Levels](#2-the-four-mean-levels)
3. [Regime Filter: DSS Bressert + Lyapunov HP](#3-regime-filter-dss-bressert--lyapunov-hp)
4. [Live System Flow](#4-live-system-flow)
5. [Backtest Results](#5-backtest-results)
6. [Installation](#6-installation)
7. [Repository Structure](#7-repository-structure)
8. [Usage](#8-usage)
9. [Configuration Reference](#9-configuration-reference)
10. [Risk Disclaimer](#10-risk-disclaimer)

---

## 1. Strategy Overview

The Mean Levels strategy treats price means — computed at multiple timeframe
resolutions — as natural support and resistance levels. Orders are placed as
limit buys at each mean level below price, letting price come to you.

A **dual-indicator regime gate** (DSS Bressert + Lyapunov HP) ensures orders
are only placed when both indicators agree on direction. This eliminates
low-conviction setups and dramatically improves signal quality.

The consolidated live system runs once daily at 9:00 AM ET:

1. Cancel all stale open orders
2. Compute regime (DSS + Lyapunov) for each ticker
3. Compute Lua-faithful mean levels (CDM, PDM, CMM, PMM)
4. Place limit buy orders at levels below price (BUY regime only)
5. Attach protective stops at each level

### v3 Backtest Highlights (18 months, 20 tickers)

| Metric | v2 (no gate) | v3 (DSS+Lyap gate) |
|---|---|---|
| Return | +91.2% | **+135.8%** |
| Trades | 894 | 1,740 |
| Win Rate | 69.1% | 58.3% |
| Profit Factor | 4.07 | 2.69 |
| Max Drawdown | 0.6% | 2.1% |

---

## 2. The Four Mean Levels

All levels use the **Lua-faithful mean algorithm**: running cumulative close
average per timeframe group (not OHLC/4 typical price).

### CDM — Current Day Mean
```
CDM = running average of Close values within the current trading day
```
Fastest-moving level. Weight: **1 pt**.

### PDM — Previous Day Mean
```
PDM = final running average of Close values from the previous trading day
```
Yesterday's pivot becomes today's first S/R reference. Weight: **2 pts**.

### CMM — Current Month Mean
```
CMM = running average of Close values within the current calendar month
```
Macro intramonth trend anchor. Weight: **3 pts**.

### PMM — Previous Month Mean
```
PMM = final running average of Close values from the prior calendar month
```
Strongest level — represents prior month's value area centre. Weight: **4 pts**.

---

## 3. Regime Filter: DSS Bressert + Lyapunov HP

Two indicators act as **hard gate filters**. If the regime is not BUY, no
orders are placed for that ticker.

### DSS Bressert (Double Smoothed Stochastic)
- Parameters: `stoch=13, smooth=8, signal=8`
- Computes double-smoothed stochastic oscillator
- Direction: BULLISH when DSS > Signal, BEARISH when DSS < Signal
- Modeled on 8-hour bars (daily bars serve as proxy for equities)

### Lyapunov HP (Hodrick-Prescott Exponent)
- Parameters: `filter=7, L_Period=525`
- Estimates Lyapunov exponent via HP-filtered returns
- Direction: BULLISH when exponent > 0 and rising, BEARISH when < 0 or falling

### Gate Logic
| DSS | Lyapunov | Regime | Action |
|-----|----------|--------|--------|
| BULLISH | BULLISH | **BUY** | Place limit orders at mean levels below price |
| BEARISH | BEARISH | SELL | Skip (can't short on Public.com) |
| Mixed / Neutral | Any | NEUTRAL | Skip (no edge) |

---

## 4. Live System Flow

`mean_levels_live.py` is the consolidated single script:

```
9:00 AM ET ──┐
             │
   ┌─────────▼──────────┐
   │ Cancel stale orders │
   └─────────┬──────────┘
             │
   ┌─────────▼──────────┐
   │ For each ticker:    │
   │  • Fetch 800d OHLCV │
   │  • Compute regime   │
   │  • Compute levels   │
   └─────────┬──────────┘
             │
   ┌─────────▼──────────┐
   │ Regime = BUY?       │
   │  Y → Limit orders   │
   │  N → Skip           │
   └─────────┬──────────┘
             │
   ┌─────────▼──────────┐
   │ Log to JSON         │
   └────────────────────┘
```

### Order Placement
- Limit BUY at each mean level below current price
- Skip levels > 5% below price (too far)
- 0.7% protective stop below each level
- Position sizing weighted by level strength (PMM=1.0×, CMM=0.75×, PDM=0.5×, CDM=0.25×)
- DAY time-in-force (orders expire at close)

### Risk Parameters
- 1% equity per trade
- Max 3 orders per ticker
- Long-only (Public.com constraint)

---

## 5. Backtest Results

### v3 Configuration (Lua-faithful + DSS/Lyap gate)

```
Months:         18
Equity:         $100,000
Risk/trade:     1% of equity
Stop-loss:      0.7% below level
Mode:           BUY regime only (DSS+Lyap gate)
Tickers:        20 (dynamic universe)
```

| Stat | Value |
|---|---|
| Total Return | +135.8% |
| Win Rate | 58.3% |
| Total Trades | 1,740 |
| Profit Factor | 2.69 |
| Max Drawdown | 2.1% |

### v2 Configuration (breaks-only, no gate)

| Stat | Value |
|---|---|
| Total Return | +91.2% |
| Win Rate | 69.1% |
| Total Trades | 894 |
| Profit Factor | 4.07 |
| Max Drawdown | 0.59% |

---

## 6. Installation

### Prerequisites
- Python 3.10 or newer
- pip
- Public.com API key (for live trading)

### Install dependencies

```bash
pip install -r requirements.txt
```

Set your Public.com API secret:
```bash
export PUBLIC_COM_SECRET="your_api_secret"
```

---

## 7. Repository Structure

```
mean-levels-strategy/
├── mean_levels_live.py          # Consolidated live system (regime → cancel → levels → orders)
├── mean_levels_scanner.py       # Standalone scanner with DSS+Lyap gate + Lua-faithful means
├── mean_levels_universe.py      # Dynamic universe selector (ranks ~110 names)
├── mean_levels_executor.py      # Legacy multi-step executor (long-only)
├── mean_levels_backtest.py      # v2 backtester (breaks/bounces, no gate)
├── mean_levels_backtest_v3.py   # v3 backtester (Lua-faithful + DSS/Lyap gate)
├── dss_bressert.py              # DSS Bressert indicator — Python port of Lua source
├── lyapunov_hp.py               # Lyapunov HP indicator — Python port of Lua source
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 8. Usage

### Consolidated Live System (Recommended)

Dry run (default) — computes everything, shows orders, places nothing:
```bash
python3 mean_levels_live.py
```

Live mode — places real orders via Public.com API:
```bash
python3 mean_levels_live.py --live
```

Specific tickers:
```bash
python3 mean_levels_live.py --live --tickers SPY QQQ AAPL NVDA
```

Override risk and max orders:
```bash
python3 mean_levels_live.py --live --risk-pct 1.5 --max-orders 4
```

JSON output:
```bash
python3 mean_levels_live.py --json
```

---

### Scanner (Standalone)

Scan default universe:
```bash
python3 mean_levels_scanner.py
```

Scan specific tickers:
```bash
python3 mean_levels_scanner.py --tickers AAPL MSFT NVDA
```

---

### Universe Selector

Print ranked universe (default top 30):
```bash
python3 mean_levels_universe.py
```

Select and immediately scan:
```bash
python3 mean_levels_universe.py --run-scanner
```

---

### Backtester (v3)

Run v3 backtest with DSS+Lyap gate:
```bash
python3 mean_levels_backtest_v3.py
```

Custom parameters:
```bash
python3 mean_levels_backtest_v3.py --tickers NVDA META TSLA --months 24 --equity 50000
```

---

## 9. Configuration Reference

### mean_levels_live.py

| Flag | Default | Description |
|---|---|---|
| `--live` | off | Place real orders (default: dry run) |
| `--tickers` | 20 defaults | Space-separated ticker symbols |
| `--risk-pct` | `1.0` | Risk per trade as % of equity |
| `--max-orders` | `3` | Max limit orders per ticker |
| `--equity` | auto | Override account equity (auto-fetched) |
| `--json` | off | Print JSON session log |

### Indicator Parameters

| Indicator | Parameter | Default | Description |
|---|---|---|---|
| DSS Bressert | stoch | 13 | Stochastic lookback period |
| DSS Bressert | smooth | 8 | Double-smoothing period |
| DSS Bressert | signal | 8 | Signal line period |
| Lyapunov HP | filter | 7 | HP filter smoothing |
| Lyapunov HP | L_Period | 525 | Lyapunov lookback period |

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
[pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/),
[matplotlib](https://matplotlib.org/), and
[publicdotcom-py](https://pypi.org/project/publicdotcom-py/).*
