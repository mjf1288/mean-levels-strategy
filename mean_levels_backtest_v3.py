#!/usr/bin/env python3
"""
Mean Levels S/R Strategy — Backtester
======================================
Walks through historical daily OHLCV data bar-by-bar, computing the 4 mean
levels on each day using ONLY data available at that point (no look-ahead),
detecting setups, simulating entries/exits, and tracking P&L.

Strategy rules (from mean_levels_strategy.pdf):
  Levels:  CDM (1pt), PDM (2pt), CMM (3pt), PMM (4pt)
  Confluence: levels within 0.5% of each other → scores add
  Min score to trade: 3

  BOUNCE_LONG: price within 0.3% of zone, stretched ≥1.5% down from 5-day
               high, closes above zone. Entry=close, stop=zone-0.5%
  BREAK_LONG:  price closes above zone after 2+ bars below it, score≥3.
               Entry=zone center, stop=zone-0.7% (wider for retest room)
  BOUNCE_SELL: mirror (triggers exit of open longs, not new shorts)
  BREAK_SELL:  mirror (triggers exit of open longs)

  Targets: T1=1R, T2=2R, T3=3R  (scale 1/3 at each)
  Stop: per-setup (0.5% for bounce, 0.7% for break below zone)
  Max loss/trade: 1% equity.  Half-size when score=3.
  Max 5 concurrent positions.

Usage:
  python3 mean_levels_backtest.py                           # Default 20 tickers, 1yr
  python3 mean_levels_backtest.py --tickers SPY QQQ NVDA    # Specific tickers
  python3 mean_levels_backtest.py --months 24               # 2-year backtest
  python3 mean_levels_backtest.py --equity 50000            # Starting equity
  python3 mean_levels_backtest.py --report                  # Generate PDF report
"""

import argparse, json, math, os, subprocess, sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ── bootstrap ───────────────────────────────────────────────────────────────
def _ensure(pkg, name=None):
    try: __import__(name or pkg)
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"--quiet"],
                              stdout=subprocess.DEVNULL)

_ensure("yfinance"); _ensure("pandas"); _ensure("matplotlib"); _ensure("numpy")

import os
import importlib.util as _ilu
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# Import directional indicators
def _import_from_file(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_WS = os.path.dirname(os.path.abspath(__file__))
_dss_mod = _import_from_file("dss_bressert", os.path.join(_WS, "dss_bressert.py"))
_lyap_mod = _import_from_file("lyapunov_hp", os.path.join(_WS, "lyapunov_hp.py"))
DSSBressert = _dss_mod.DSSBressert
LyapunovHP = _lyap_mod.LyapunovHP

_DSS = DSSBressert()
_LYAP = LyapunovHP()

# ── colours ─────────────────────────────────────────────────────────────────
G="\033[92m"; R="\033[91m"; Y="\033[93m"; C="\033[96m"; B="\033[1m"; X="\033[0m"
def green(s): return f"{G}{s}{X}"
def red(s):   return f"{R}{s}{X}"
def yellow(s):return f"{Y}{s}{X}"
def cyan(s):  return f"{C}{s}{X}"
def bold(s):  return f"{B}{s}{X}"

# ── constants ───────────────────────────────────────────────────────────────
LEVEL_WEIGHTS = {"PMM":4, "CMM":3, "PDM":2, "CDM":1}
DEFAULT_TICKERS = [
    "SPY","QQQ","IWM","AAPL","MSFT","NVDA","GOOGL","META",
    "AMZN","TSLA","AMD","NFLX","CRM","AVGO","JPM","V",
    "MA","UNH","LLY","COST",
]
RESULTS_DIR = "/home/user/workspace/backtest_results"

# ── data classes ────────────────────────────────────────────────────────────
@dataclass
class Trade:
    ticker: str
    setup_type: str        # BOUNCE_LONG | BREAK_LONG
    entry_date: str
    entry_price: float
    stop_price: float
    t1: float
    t2: float
    t3: float
    score: int
    levels: List[str]
    shares_initial: int
    shares_remaining: int = 0
    exit_date: str = ""
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    outcome: str = ""       # WIN | LOSS | PARTIAL
    hold_days: int = 0
    partial_exits: List[dict] = field(default_factory=list)

    def __post_init__(self):
        self.shares_remaining = self.shares_initial

# ── mean levels (Lua-faithful, no look-ahead) ─────────────────────────────
def _calc_mean_engine(closes, groups):
    """Lua-faithful running cumulative close average per group."""
    cum, count = 0.0, 0
    buff, prev_buff, prev_tf_buff = 0.0, 0.0, None
    direction = 0
    current_group = None
    for i in range(len(closes)):
        src = float(closes[i])
        grp = groups[i]
        if i == 0 or grp != current_group:
            if i > 0:
                prev_tf_buff = buff
            cum, count, current_group = src, 1, grp
        else:
            cum += src
            count += 1
        prev_buff = buff
        buff = cum / count
        if buff > prev_buff: direction = 1
        elif buff < prev_buff: direction = -1
    return {
        "current": round(buff, 6),
        "prev": round(prev_tf_buff, 6) if prev_tf_buff is not None else round(buff, 6),
        "dir": "UP" if direction == 1 else "DOWN",
    }


def compute_levels_at(df: pd.DataFrame, idx: int) -> Optional[Dict[str, float]]:
    """Compute the 4 mean levels using Lua-faithful engine. No look-ahead."""
    if idx < 1:
        return None
    sub = df.iloc[:idx + 1]  # only data up to and including bar[idx]
    closes = sub["Close"].tolist()
    # Daily groups
    daily_groups = [
        (ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date())
        for ts in sub.index
    ]
    daily = _calc_mean_engine(closes, daily_groups)
    cdm = daily["current"]
    pdm = daily["prev"]
    # Monthly groups
    monthly_groups = [
        (ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()).strftime("%Y-%m")
        for ts in sub.index
    ]
    monthly = _calc_mean_engine(closes, monthly_groups)
    cmm = monthly["current"]
    pmm = monthly["prev"]
    return {"CDM": cdm, "PDM": pdm, "CMM": cmm, "PMM": pmm}


def confluence_zones(levels: Dict[str,float], prox: float = 0.005) -> List[Dict]:
    items = [(v, k, LEVEL_WEIGHTS[k]) for k, v in levels.items()]
    assigned = [False]*len(items)
    zones = []
    for i,(pi,ni,wi) in enumerate(items):
        if assigned[i]: continue
        gp, gn, gw, gm = [pi],[ni],[wi],{ni:pi}
        assigned[i] = True
        for j,(pj,nj,wj) in enumerate(items):
            if i==j or assigned[j]: continue
            if abs(pi-pj)/max(pi,pj) <= prox:
                gp.append(pj); gn.append(nj); gw.append(wj); gm[nj]=pj; assigned[j]=True
        zones.append({"center":sum(gp)/len(gp),"score":sum(gw),"levels":gn,"prices":gm})
    zones.sort(key=lambda z: z["score"], reverse=True)
    return zones


# ── setup detection (per bar) ───────────────────────────────────────────────
def detect_setups_at(df: pd.DataFrame, idx: int, levels: Dict[str,float],
                     zones: List[Dict]) -> List[Dict]:
    """Detect setups at bar[idx]. No future data used."""
    if idx < 5:
        return []

    price = float(df.iloc[idx]["Close"])
    hi5   = float(df.iloc[max(0,idx-4):idx+1]["High"].max())
    lo5   = float(df.iloc[max(0,idx-4):idx+1]["Low"].min())
    stretch_dn = (hi5 - price) / hi5
    stretch_up = (price - lo5) / lo5 if lo5 > 0 else 0

    setups = []
    for zone in zones:
        c, sc = zone["center"], zone["score"]
        if sc < 3:
            continue
        prox = abs(price - c) / price

        # BOUNCE_LONG
        if prox <= 0.003 and stretch_dn >= 0.015 and price >= c:
            stop = c * 0.995
            risk = abs(price - stop)
            if risk > 0:
                setups.append({"type":"BOUNCE_LONG","zone":zone,"score":sc,
                    "entry":price,"stop":stop,"t1":price+risk,"t2":price+2*risk,"t3":price+3*risk,
                    "risk":risk})

        # BREAK_LONG
        if idx >= 2 and price > c and sc >= 3:
            below = sum(1 for k in [idx-1, idx-2] if float(df.iloc[k]["Close"]) < c)
            if below >= 2:
                stop = c * 0.993
                risk = abs(c - stop)
                if risk > 0:
                    setups.append({"type":"BREAK_LONG","zone":zone,"score":sc,
                        "entry":c,"stop":stop,"t1":c+risk,"t2":c+2*risk,"t3":c+3*risk,
                        "risk":risk})

        # BOUNCE_SELL / BREAK_SELL → used to EXIT open longs, not open new positions
        # We tag them so the position manager knows to exit
        if prox <= 0.003 and stretch_up >= 0.015 and price <= c and sc >= 3:
            setups.append({"type":"BOUNCE_SELL","zone":zone,"score":sc,"center":c})
        if idx >= 2 and price < c and sc >= 3:
            above = sum(1 for k in [idx-1, idx-2] if float(df.iloc[k]["Close"]) > c)
            if above >= 2:
                setups.append({"type":"BREAK_SELL","zone":zone,"score":sc,"center":c})

    return setups


# ── position sizing ─────────────────────────────────────────────────────────
def size_position(equity: float, entry: float, stop: float, score: int,
                  risk_pct: float = 0.02) -> int:
    risk_per = abs(entry - stop)
    if risk_per == 0: return 0
    dollars = equity * risk_pct
    if score <= 3: dollars *= 0.5
    return max(int(dollars / risk_per), 1)


# ── single-ticker backtest ──────────────────────────────────────────────────
def backtest_ticker(ticker: str, df: pd.DataFrame, starting_equity: float,
                    max_positions: int = 5) -> Tuple[List[Trade], pd.Series]:
    """
    Walk forward through *df* bar-by-bar. Track equity, open/close trades.
    Returns (list_of_trades, daily_equity_series).
    """
    equity = starting_equity
    trades_closed: List[Trade] = []
    trades_open:   List[Trade] = []
    equity_curve = {}

    for idx in range(5, len(df)):
        bar   = df.iloc[idx]
        dt    = str(pd.Timestamp(bar.name).date())
        price = float(bar["Close"])
        high  = float(bar["High"])
        low   = float(bar["Low"])

        levels = compute_levels_at(df, idx)
        if levels is None:
            equity_curve[dt] = equity
            continue
        zones = confluence_zones(levels)
        setups = detect_setups_at(df, idx, levels, zones)

        sell_signals = [s for s in setups if s["type"] in ("BOUNCE_SELL","BREAK_SELL")]
        buy_signals  = [s for s in setups if s["type"] in ("BREAK_LONG",)]  # bounce filtered out

        # ── directional gate (DSS + Lyapunov) ─────────────────────────
        sub = df.iloc[:idx + 1]
        dss_result = _DSS.calculate(sub)
        lyap_result = _LYAP.calculate_from_df(sub)
        dss_dir = dss_result["direction"] if dss_result else "NEUTRAL"
        lyap_dir = lyap_result["direction"] if lyap_result else "NEUTRAL"
        if dss_dir == "BULLISH" and lyap_dir == "BULLISH":
            net_dir = "BULLISH"
        elif dss_dir == "BEARISH" and lyap_dir == "BEARISH":
            net_dir = "BEARISH"
        elif dss_dir == "NEUTRAL" and lyap_dir != "NEUTRAL":
            net_dir = lyap_dir
        elif lyap_dir == "NEUTRAL" and dss_dir != "NEUTRAL":
            net_dir = dss_dir
        else:
            net_dir = "NEUTRAL"
        if net_dir == "BEARISH":
            buy_signals = []  # gate kills all long entries when bearish


        # ── manage open positions ───────────────────────────────────────
        still_open = []
        for t in trades_open:
            closed = False

            # Check stop hit (use low of bar)
            if low <= t.stop_price:
                exit_p = t.stop_price
                pnl = (exit_p - t.entry_price) * t.shares_remaining
                for pe in t.partial_exits:
                    pnl += pe["pnl"]
                t.exit_date = dt; t.exit_price = exit_p
                t.pnl = round(pnl, 2)
                t.pnl_pct = round(pnl / (t.entry_price * t.shares_initial) * 100, 2)
                t.outcome = "LOSS" if t.pnl < 0 else "WIN"
                t.hold_days = (pd.Timestamp(dt) - pd.Timestamp(t.entry_date)).days
                equity += pnl
                trades_closed.append(t)
                closed = True

            # Check sell signal → exit remaining
            if not closed and sell_signals:
                exit_p = price
                pnl = (exit_p - t.entry_price) * t.shares_remaining
                for pe in t.partial_exits:
                    pnl += pe["pnl"]
                t.exit_date = dt; t.exit_price = exit_p
                t.pnl = round(pnl, 2)
                t.pnl_pct = round(pnl / (t.entry_price * t.shares_initial) * 100, 2)
                t.outcome = "WIN" if t.pnl > 0 else "LOSS"
                t.hold_days = (pd.Timestamp(dt) - pd.Timestamp(t.entry_date)).days
                equity += pnl
                trades_closed.append(t)
                closed = True

            # Partial profit-taking
            if not closed:
                # T1 hit → scale out 1/3, move stop to breakeven
                if high >= t.t1 and t.shares_remaining == t.shares_initial:
                    scale_out = max(t.shares_remaining // 3, 1)
                    partial_pnl = (t.t1 - t.entry_price) * scale_out
                    t.shares_remaining -= scale_out
                    t.partial_exits.append({"target":"T1","price":t.t1,"shares":scale_out,"pnl":round(partial_pnl,2)})
                    t.stop_price = t.entry_price   # move stop to BE
                    equity += partial_pnl

                # T2 hit → scale out another 1/3
                if high >= t.t2 and t.shares_remaining > t.shares_initial // 3:
                    scale_out = max(t.shares_remaining // 2, 1)
                    partial_pnl = (t.t2 - t.entry_price) * scale_out
                    t.shares_remaining -= scale_out
                    t.partial_exits.append({"target":"T2","price":t.t2,"shares":scale_out,"pnl":round(partial_pnl,2)})
                    t.stop_price = t.t1   # trail stop to T1
                    equity += partial_pnl

                # T3 hit → exit remaining
                if high >= t.t3 and t.shares_remaining > 0:
                    partial_pnl = (t.t3 - t.entry_price) * t.shares_remaining
                    t.partial_exits.append({"target":"T3","price":t.t3,"shares":t.shares_remaining,"pnl":round(partial_pnl,2)})
                    total_pnl = sum(pe["pnl"] for pe in t.partial_exits)
                    t.exit_date = dt; t.exit_price = t.t3
                    t.pnl = round(total_pnl, 2)
                    t.pnl_pct = round(total_pnl / (t.entry_price * t.shares_initial) * 100, 2)
                    t.outcome = "WIN"
                    t.hold_days = (pd.Timestamp(dt) - pd.Timestamp(t.entry_date)).days
                    equity += (t.t3 - t.entry_price) * (t.shares_remaining + sum(pe["shares"] for pe in t.partial_exits[:-1]) - t.shares_initial + t.shares_remaining)
                    # Simpler: just add the T3 partial pnl
                    equity_adj = partial_pnl  # we already added T1/T2 partials above
                    # equity already accumulated partials; just track closure
                    trades_closed.append(t)
                    closed = True

                # Time stop: close after 10 days
                hold = (pd.Timestamp(dt) - pd.Timestamp(t.entry_date)).days
                if not closed and hold >= 10:
                    exit_p = price
                    remaining_pnl = (exit_p - t.entry_price) * t.shares_remaining
                    total_pnl = remaining_pnl + sum(pe["pnl"] for pe in t.partial_exits)
                    t.exit_date = dt; t.exit_price = exit_p
                    t.pnl = round(total_pnl, 2)
                    t.pnl_pct = round(total_pnl / (t.entry_price * t.shares_initial) * 100, 2)
                    t.outcome = "WIN" if t.pnl > 0 else "LOSS"
                    t.hold_days = hold
                    equity += remaining_pnl
                    trades_closed.append(t)
                    closed = True

            if not closed:
                still_open.append(t)

        trades_open = still_open

        # ── open new positions ──────────────────────────────────────────
        if len(trades_open) < max_positions:
            # Already holding this ticker?
            held_tickers = {t.ticker for t in trades_open}
            for sig in sorted(buy_signals, key=lambda s: s["score"], reverse=True):
                if len(trades_open) >= max_positions:
                    break
                if ticker in held_tickers:
                    break

                shares = size_position(equity, sig["entry"], sig["stop"], sig["score"])
                if shares == 0:
                    continue

                trade = Trade(
                    ticker=ticker,
                    setup_type=sig["type"],
                    entry_date=dt,
                    entry_price=round(sig["entry"], 4),
                    stop_price=round(sig["stop"], 4),
                    t1=round(sig["t1"], 4),
                    t2=round(sig["t2"], 4),
                    t3=round(sig["t3"], 4),
                    score=sig["score"],
                    levels=sig["zone"]["levels"],
                    shares_initial=shares,
                )
                trades_open.append(trade)
                held_tickers.add(ticker)
                break   # one entry per ticker per bar

        equity_curve[dt] = round(equity, 2)

    # Close any remaining open trades at last bar's close
    last_price = float(df.iloc[-1]["Close"])
    last_dt    = str(pd.Timestamp(df.iloc[-1].name).date())
    for t in trades_open:
        remaining_pnl = (last_price - t.entry_price) * t.shares_remaining
        total_pnl = remaining_pnl + sum(pe["pnl"] for pe in t.partial_exits)
        t.exit_date = last_dt; t.exit_price = last_price
        t.pnl = round(total_pnl, 2)
        t.pnl_pct = round(total_pnl / (t.entry_price * t.shares_initial) * 100, 2)
        t.outcome = "WIN" if t.pnl > 0 else "LOSS"
        t.hold_days = (pd.Timestamp(last_dt) - pd.Timestamp(t.entry_date)).days
        equity += remaining_pnl
        trades_closed.append(t)
    equity_curve[last_dt] = round(equity, 2)

    eq_series = pd.Series(equity_curve, dtype=float)
    eq_series.index = pd.to_datetime(eq_series.index)
    return trades_closed, eq_series


# ── portfolio-level runner ──────────────────────────────────────────────────
def run_backtest(tickers: List[str], months: int = 12, equity: float = 100_000) -> Dict:
    # Need extra data for Lyapunov warmup (525 bars) + DSS warmup (~50 bars)
    warmup_days = 800  # ~3 years of trading days covers L_Period=525
    actual_days = int(months * 30.5)
    total_days = max(actual_days, warmup_days)
    period = f"{total_days}d"
    print(cyan(f"\nDownloading {len(tickers)} tickers ({months}mo + warmup = {total_days}d) …"))

    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)

    all_trades: List[Trade] = []
    ticker_stats = {}

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:>2}/{len(tickers)}] {ticker:<7}", end=" ", flush=True)
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()
            df.dropna(subset=["Close","High","Low","Open","Volume"], inplace=True)
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_convert("America/New_York").tz_localize(None)
            df.sort_index(inplace=True)

            if len(df) < 25:
                print(yellow("skipped (insufficient data)"))
                continue

            trades, eq = backtest_ticker(ticker, df, equity / len(tickers))
            n_trades = len(trades)
            wins = sum(1 for t in trades if t.outcome == "WIN")
            total_pnl = sum(t.pnl for t in trades)
            label = green(f"{n_trades} trades, PnL ${total_pnl:+,.0f}") if total_pnl >= 0 else red(f"{n_trades} trades, PnL ${total_pnl:+,.0f}")
            print(label)

            all_trades.extend(trades)
            ticker_stats[ticker] = {
                "trades": n_trades,
                "wins": wins,
                "win_rate": round(wins/n_trades*100, 1) if n_trades else 0,
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(total_pnl/n_trades, 2) if n_trades else 0,
            }
        except Exception as exc:
            print(yellow(f"error — {exc}"))
            continue

    return {
        "tickers": tickers,
        "months": months,
        "starting_equity": equity,
        "trades": all_trades,
        "ticker_stats": ticker_stats,
    }


# ── analytics ───────────────────────────────────────────────────────────────
def compute_analytics(result: Dict) -> Dict:
    trades = result["trades"]
    equity = result["starting_equity"]
    if not trades:
        return {"error": "No trades generated"}

    pnls     = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    wins     = [t for t in trades if t.outcome == "WIN"]
    losses   = [t for t in trades if t.outcome == "LOSS"]
    bounces  = [t for t in trades if "BOUNCE" in t.setup_type]
    breaks   = [t for t in trades if "BREAK" in t.setup_type]
    holds    = [t.hold_days for t in trades]

    total_pnl    = sum(pnls)
    win_rate     = len(wins)/len(trades)*100 if trades else 0
    avg_win      = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss     = np.mean([t.pnl for t in losses]) if losses else 0
    profit_factor= abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')
    expectancy   = np.mean(pnls) if pnls else 0

    # Build daily equity curve from trade dates
    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t.exit_date] += t.pnl
    dates_sorted = sorted(daily_pnl.keys())
    eq_curve = []
    running = equity
    for d in dates_sorted:
        running += daily_pnl[d]
        eq_curve.append({"date": d, "equity": round(running, 2)})

    # Max drawdown
    peak = equity
    max_dd = 0
    max_dd_pct = 0
    for pt in eq_curve:
        if pt["equity"] > peak:
            peak = pt["equity"]
        dd = peak - pt["equity"]
        dd_pct = dd / peak * 100
        if dd_pct > max_dd_pct:
            max_dd = dd
            max_dd_pct = dd_pct

    # By score tier
    score_tiers = defaultdict(lambda: {"count":0,"wins":0,"pnl":0})
    for t in trades:
        tier = "High (≥5)" if t.score >= 5 else "Moderate (3-4)"
        score_tiers[tier]["count"] += 1
        score_tiers[tier]["pnl"] += t.pnl
        if t.outcome == "WIN": score_tiers[tier]["wins"] += 1

    # Monthly returns
    monthly = defaultdict(float)
    for t in trades:
        m = t.exit_date[:7]
        monthly[m] += t.pnl

    return {
        "total_trades":     len(trades),
        "total_pnl":        round(total_pnl, 2),
        "total_return_pct": round(total_pnl / equity * 100, 2),
        "win_rate":         round(win_rate, 1),
        "wins":             len(wins),
        "losses":           len(losses),
        "avg_win":          round(avg_win, 2),
        "avg_loss":         round(avg_loss, 2),
        "profit_factor":    round(profit_factor, 2),
        "expectancy":       round(expectancy, 2),
        "max_drawdown":     round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "avg_hold_days":    round(np.mean(holds), 1) if holds else 0,
        "bounce_trades":    len(bounces),
        "break_trades":     len(breaks),
        "bounce_win_rate":  round(sum(1 for t in bounces if t.outcome=="WIN")/len(bounces)*100,1) if bounces else 0,
        "break_win_rate":   round(sum(1 for t in breaks if t.outcome=="WIN")/len(breaks)*100,1) if breaks else 0,
        "bounce_pnl":       round(sum(t.pnl for t in bounces), 2),
        "break_pnl":        round(sum(t.pnl for t in breaks), 2),
        "score_tiers":      dict(score_tiers),
        "monthly_pnl":      dict(monthly),
        "equity_curve":     eq_curve,
        "by_ticker":        result["ticker_stats"],
    }


# ── charting ────────────────────────────────────────────────────────────────
PALETTE = ['#20808D','#A84B2F','#1B474D','#BCE2E7','#944454','#FFC553','#848456','#6E522B']

def generate_charts(analytics: Dict, result: Dict, output_dir: str = RESULTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    trades = result["trades"]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'figure.dpi':150,'font.size':11,'axes.titleweight':'bold'})

    # 1. Equity curve
    eq = analytics["equity_curve"]
    if eq:
        fig, ax = plt.subplots(figsize=(12, 5))
        dates = [pd.Timestamp(e["date"]) for e in eq]
        vals  = [e["equity"] for e in eq]
        ax.plot(dates, vals, color=PALETTE[0], linewidth=2)
        ax.axhline(result["starting_equity"], color="#999", linestyle="--", linewidth=1, label="Starting equity")
        ax.fill_between(dates, result["starting_equity"], vals,
                        where=[v >= result["starting_equity"] for v in vals],
                        alpha=0.15, color=PALETTE[0])
        ax.fill_between(dates, result["starting_equity"], vals,
                        where=[v < result["starting_equity"] for v in vals],
                        alpha=0.15, color=PALETTE[1])
        ax.set_title(f"Equity Curve — ${result['starting_equity']:,.0f} → ${vals[-1]:,.0f}  ({analytics['total_return_pct']:+.1f}%)")
        ax.set_ylabel("Portfolio Equity ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"${x:,.0f}"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.legend()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/equity_curve.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 2. P&L distribution
    if trades:
        fig, ax = plt.subplots(figsize=(10, 5))
        pnls = [t.pnl for t in trades]
        colors = [PALETTE[0] if p >= 0 else PALETTE[1] for p in pnls]
        ax.hist(pnls, bins=min(40, max(10, len(pnls)//3)), color=PALETTE[0], edgecolor='white', alpha=0.8)
        ax.axvline(0, color='#333', linewidth=1.5, linestyle='-')
        mean_pnl = np.mean(pnls)
        ax.axvline(mean_pnl, color=PALETTE[1], linewidth=1.5, linestyle='--',
                   label=f"Mean: ${mean_pnl:,.0f}")
        ax.set_title(f"P&L Distribution — {len(trades)} Trades")
        ax.set_xlabel("Trade P&L ($)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pnl_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Win rate by setup type
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: win rate comparison
    types = ["BOUNCE_LONG", "BREAK_LONG"]
    wr = []
    counts = []
    for st in types:
        subset = [t for t in trades if t.setup_type == st]
        w = sum(1 for t in subset if t.outcome == "WIN")
        wr.append(w/len(subset)*100 if subset else 0)
        counts.append(len(subset))

    bars = axes[0].bar(types, wr, color=[PALETTE[0], PALETTE[2]], width=0.5)
    for bar, c, w in zip(bars, counts, wr):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{w:.0f}%\n({c})", ha='center', va='bottom', fontsize=10)
    axes[0].set_title("Win Rate by Setup Type")
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].set_ylim(0, 100)
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

    # Right: P&L by setup type
    pnl_by_type = {}
    for st in types:
        pnl_by_type[st] = sum(t.pnl for t in trades if t.setup_type == st)
    colors_bar = [PALETTE[0] if v >= 0 else PALETTE[1] for v in pnl_by_type.values()]
    bars2 = axes[1].bar(pnl_by_type.keys(), pnl_by_type.values(), color=colors_bar, width=0.5)
    for bar, v in zip(bars2, pnl_by_type.values()):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height() + (50 if v >=0 else -50),
                    f"${v:,.0f}", ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
    axes[1].set_title("Total P&L by Setup Type")
    axes[1].set_ylabel("P&L ($)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"${x:,.0f}"))
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/setup_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Monthly returns heatmap-bar
    if analytics["monthly_pnl"]:
        fig, ax = plt.subplots(figsize=(12, 4))
        months_sorted = sorted(analytics["monthly_pnl"].keys())
        vals = [analytics["monthly_pnl"][m] for m in months_sorted]
        colors_m = [PALETTE[0] if v >= 0 else PALETTE[1] for v in vals]
        ax.bar(months_sorted, vals, color=colors_m, width=0.7)
        ax.set_title("Monthly P&L")
        ax.set_ylabel("P&L ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"${x:,.0f}"))
        ax.axhline(0, color='#333', linewidth=0.8)
        plt.xticks(rotation=45, ha='right')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/monthly_pnl.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Score tier analysis
    if analytics["score_tiers"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        tiers = list(analytics["score_tiers"].keys())
        tier_wr = [analytics["score_tiers"][t]["wins"]/analytics["score_tiers"][t]["count"]*100
                   if analytics["score_tiers"][t]["count"] > 0 else 0 for t in tiers]
        tier_counts = [analytics["score_tiers"][t]["count"] for t in tiers]
        bars = ax.bar(tiers, tier_wr, color=[PALETTE[0], PALETTE[2]][:len(tiers)], width=0.5)
        for bar, c, w in zip(bars, tier_counts, tier_wr):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                   f"{w:.0f}%\n({c} trades)", ha='center', fontsize=10)
        ax.set_title("Win Rate by Confluence Score")
        ax.set_ylabel("Win Rate (%)")
        ax.set_ylim(0, 100)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/score_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 6. Per-ticker performance
    if analytics["by_ticker"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        tickers_sorted = sorted(analytics["by_ticker"].keys(),
                               key=lambda t: analytics["by_ticker"][t]["total_pnl"])
        pnls_t = [analytics["by_ticker"][t]["total_pnl"] for t in tickers_sorted]
        colors_t = [PALETTE[0] if v >= 0 else PALETTE[1] for v in pnls_t]
        ax.barh(tickers_sorted, pnls_t, color=colors_t, height=0.6)
        for i, (t, v) in enumerate(zip(tickers_sorted, pnls_t)):
            wr = analytics["by_ticker"][t]["win_rate"]
            n = analytics["by_ticker"][t]["trades"]
            ax.text(v + (50 if v >= 0 else -50), i,
                   f"${v:,.0f} ({wr:.0f}% WR, {n}t)", va='center',
                   ha='left' if v >= 0 else 'right', fontsize=9)
        ax.set_title("P&L by Ticker")
        ax.set_xlabel("Total P&L ($)")
        ax.axvline(0, color='#333', linewidth=0.8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"${x:,.0f}"))
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ticker_performance.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nCharts saved → {output_dir}/")


# ── console report ──────────────────────────────────────────────────────────
def print_report(analytics: Dict, result: Dict):
    a = analytics
    e = result["starting_equity"]
    print()
    print(bold("=" * 80))
    print(bold("  MEAN LEVELS v3 — BACKTEST (Lua-faithful + DSS/Lyapunov gate)"))
    print(bold("=" * 80))
    print()
    print(f"  Period:          {result['months']} months")
    print(f"  Tickers:         {len(result['tickers'])} ({', '.join(result['tickers'][:10])}{'…' if len(result['tickers'])>10 else ''})")
    print(f"  Starting equity: ${e:,.0f}")
    print()

    pnl_color = green if a["total_pnl"] >= 0 else red
    print(bold("  PERFORMANCE"))
    total_pnl_str = f"${a['total_pnl']:+,.2f}"
    print(f"  Total P&L:       {pnl_color(total_pnl_str)}")
    dd_str = f"${a['max_drawdown']:,.2f} ({a['max_drawdown_pct']:.1f}%)"
    print(f"  Max drawdown:    {red(dd_str)}")
    print()

    print(bold("  TRADE STATISTICS"))
    print(f"  Total trades:    {a['total_trades']}")
    print(f"  Win rate:        {a['win_rate']:.1f}% ({a['wins']}W / {a['losses']}L)")
    avg_win_str = f"${a['avg_win']:+,.2f}"
    print(f"  Avg win:         {green(avg_win_str)}")
    avg_loss_str = f"${a['avg_loss']:+,.2f}"
    print(f"  Avg loss:        {red(avg_loss_str)}")
    print(f"  Profit factor:   {a['profit_factor']:.2f}")
    print(f"  Expectancy:      ${a['expectancy']:+,.2f} per trade")
    print(f"  Avg hold:        {a['avg_hold_days']:.1f} days")
    print()

    print(bold("  SETUP TYPE BREAKDOWN"))
    print(f"  Bounce trades:   {a['bounce_trades']} ({a['bounce_win_rate']:.0f}% WR, PnL ${a['bounce_pnl']:+,.2f})")
    print(f"  Break trades:    {a['break_trades']} ({a['break_win_rate']:.0f}% WR, PnL ${a['break_pnl']:+,.2f})")
    print()

    print(bold("  CONFLUENCE SCORE TIERS"))
    for tier, stats in sorted(a["score_tiers"].items()):
        wr = stats["wins"]/stats["count"]*100 if stats["count"] else 0
        print(f"  {tier:<18} {stats['count']:>3} trades | {wr:>5.1f}% WR | PnL ${stats['pnl']:>+10,.2f}")
    print()

    print(bold("  TOP / BOTTOM TICKERS"))
    sorted_t = sorted(a["by_ticker"].items(), key=lambda x: x[1]["total_pnl"], reverse=True)
    for t, s in sorted_t[:5]:
        print(f"  {green(f'{t:<7}')}  {s['trades']:>2}t  {s['win_rate']:>5.1f}% WR  PnL ${s['total_pnl']:>+10,.2f}")
    if len(sorted_t) > 5:
        print("  ---")
        for t, s in sorted_t[-3:]:
            print(f"  {red(f'{t:<7}')}  {s['trades']:>2}t  {s['win_rate']:>5.1f}% WR  PnL ${s['total_pnl']:>+10,.2f}")
    print()
    print(bold("=" * 80))


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", metavar="T",
                        help="Tickers to backtest (default: 20 liquid names)")
    parser.add_argument("--months", type=int, default=12, help="Months of history (default: 12)")
    parser.add_argument("--equity", type=float, default=100_000, help="Starting equity (default: 100000)")
    parser.add_argument("--report", action="store_true", help="Generate chart images")
    parser.add_argument("--json", action="store_true", help="JSON output of analytics")

    args = parser.parse_args()
    tickers = args.tickers or DEFAULT_TICKERS

    result = run_backtest(tickers, months=args.months, equity=args.equity)
    analytics = compute_analytics(result)

    if args.json:
        # Strip non-serializable
        out = {k: v for k, v in analytics.items() if k != "equity_curve"}
        print(json.dumps(out, indent=2, default=str))
    else:
        print_report(analytics, result)

    if args.report or not args.json:
        generate_charts(analytics, result)

    # Save full results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trades_out = [asdict(t) for t in result["trades"]]
    with open(f"{RESULTS_DIR}/trades.json", "w") as f:
        json.dump(trades_out, f, indent=2, default=str)
    with open(f"{RESULTS_DIR}/analytics.json", "w") as f:
        json.dump(analytics, f, indent=2, default=str)
    print(f"\nDetailed results → {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
