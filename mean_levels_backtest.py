"""
mean_levels_backtest.py
=======================
Backtesting Engine for the Mean Levels S/R Strategy
-----------------------------------------------------
Simulates the Mean Levels trading strategy on historical daily data using
the same CDM/PDM/CMM/PMM confluence logic as the live scanner.

Entry Logic
-----------
  BREAK entries (default active):
    - BREAK_LONG  : next open after price closes above a confluence zone
    - BREAK_SHORT : next open after price closes below a confluence zone

  BOUNCE entries (optional, use --all-setups to enable):
    - BOUNCE_LONG  : price pulls back to zone from above
    - BOUNCE_SHORT : price pulls back to zone from below

Position Sizing
---------------
  Fixed-fractional risk: default 1% of current equity per trade.
  Stop-loss  = 1× ATR below (long) or above (short) the entry price.
  Take-profit = entry ± 2× risk (R:R = 1:2).

Trade Management
----------------
  Simulated on end-of-day bar closes (next open approximation).
  One position per ticker at a time.
  Max concurrent positions: configurable (default unlimited).

Metrics Reported
----------------
  Total return, CAGR, win rate, profit factor, max drawdown,
  expectancy, average hold time (bars), Sharpe ratio (approximate),
  breakdown by long/short and break/bounce.

Output
------
  backtest_results/trades.json    – trade-by-trade log
  backtest_results/summary.json   – aggregate metrics
  Console table (default)

Usage
-----
  python3 mean_levels_backtest.py
  python3 mean_levels_backtest.py --tickers NVDA META TSLA --months 24 --equity 50000
  python3 mean_levels_backtest.py --breaks-only --risk-pct 0.02
  python3 mean_levels_backtest.py --min-score 5 --json
  python3 mean_levels_backtest.py --all-setups --risk-pct 0.01
"""

import argparse
import json
import os
import sys
from datetime import datetime, date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Add parent dir to path so we can import scanner helpers
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from mean_levels_scanner import (
    compute_mean_levels,
    find_confluence_zones,
    compute_atr,
    DEFAULT_TICKERS,
    DEFAULT_CONFLUENCE_PCT,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_EQUITY: float = 100_000.0
DEFAULT_RISK_PCT: float = 0.01        # 1 % of equity per trade
DEFAULT_MONTHS: int = 12
DEFAULT_MIN_SCORE: int = 3            # minimum confluence score to trade
DEFAULT_ATR_STOP_MULT: float = 1.0   # stop = entry ± N × ATR
DEFAULT_RR: float = 2.0              # take-profit = entry ± RR × risk

OUTPUT_DIR: str = "backtest_results"


# ---------------------------------------------------------------------------
# Historical data
# ---------------------------------------------------------------------------

def fetch_history(ticker: str, months: int) -> pd.DataFrame:
    """Download daily OHLCV data for *months* months of history."""
    period = f"{months}mo" if months <= 24 else "max"
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
    except Exception as exc:
        raise ValueError(f"Download failed for {ticker}: {exc}") from exc

    if df.empty:
        raise ValueError(f"No data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.dropna(subset=["Close"], inplace=True)
    return df.sort_index()


# ---------------------------------------------------------------------------
# Walk-forward mean level computation (point-in-time safe)
# ---------------------------------------------------------------------------

def get_levels_at_bar(df: pd.DataFrame, idx: int) -> dict[str, float | None]:
    """Compute mean levels using only data available at bar index *idx*.

    This is point-in-time safe – it never looks forward.
    """
    history = df.iloc[: idx + 1].copy()
    return compute_mean_levels(history)


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

class Position:
    """Represents an open simulated position."""

    __slots__ = (
        "ticker", "direction", "entry_date", "entry_price",
        "stop", "target", "shares", "risk_dollars", "setup_type",
        "zone_price", "zone_score",
    )

    def __init__(
        self,
        ticker: str,
        direction: str,
        entry_date: date,
        entry_price: float,
        stop: float,
        target: float,
        shares: float,
        risk_dollars: float,
        setup_type: str,
        zone_price: float,
        zone_score: int,
    ) -> None:
        self.ticker = ticker
        self.direction = direction          # 'LONG' | 'SHORT'
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.stop = stop
        self.target = target
        self.shares = shares
        self.risk_dollars = risk_dollars
        self.setup_type = setup_type
        self.zone_price = zone_price
        self.zone_score = zone_score


def simulate_ticker(
    ticker: str,
    df: pd.DataFrame,
    initial_equity: float,
    risk_pct: float,
    atr_stop_mult: float,
    rr: float,
    breaks_only: bool,
    min_score: int,
    confluence_pct: float,
) -> list[dict[str, Any]]:
    """Walk-forward simulation for a single ticker.

    Returns a list of completed trade records.
    """
    trades: list[dict[str, Any]] = []
    equity = initial_equity
    position: Position | None = None

    # We need at least 40 bars of warm-up for monthly means
    warm_up = 40

    for i in range(warm_up, len(df) - 1):
        bar = df.iloc[i]
        next_bar = df.iloc[i + 1]
        bar_date = df.index[i].date()
        next_date = df.index[i + 1].date()

        # --- Manage open position ---
        if position is not None:
            open_p = float(next_bar["Open"])
            close_p = float(next_bar["Close"])
            high_p = float(next_bar["High"])
            low_p = float(next_bar["Low"])

            exit_price: float | None = None
            exit_reason: str = ""

            if position.direction == "LONG":
                if low_p <= position.stop:
                    exit_price = position.stop
                    exit_reason = "STOP"
                elif high_p >= position.target:
                    exit_price = position.target
                    exit_reason = "TARGET"
            else:  # SHORT
                if high_p >= position.stop:
                    exit_price = position.stop
                    exit_reason = "STOP"
                elif low_p <= position.target:
                    exit_price = position.target
                    exit_reason = "TARGET"

            if exit_price is not None:
                pnl = (exit_price - position.entry_price) * position.shares
                if position.direction == "SHORT":
                    pnl = -pnl
                equity += pnl
                hold_bars = (next_date - position.entry_date).days

                trades.append({
                    "ticker": ticker,
                    "direction": position.direction,
                    "setup_type": position.setup_type,
                    "zone_price": round(position.zone_price, 4),
                    "zone_score": position.zone_score,
                    "entry_date": position.entry_date.isoformat(),
                    "exit_date": next_date.isoformat(),
                    "entry_price": round(position.entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "stop": round(position.stop, 4),
                    "target": round(position.target, 4),
                    "shares": round(position.shares, 4),
                    "risk_dollars": round(position.risk_dollars, 2),
                    "pnl": round(pnl, 2),
                    "exit_reason": exit_reason,
                    "hold_bars": hold_bars,
                    "win": pnl > 0,
                    "equity_after": round(equity, 2),
                })
                position = None

            # If still open, continue to next bar
            if position is not None:
                continue

        # --- Look for new entry signals ---
        if position is not None:
            continue  # already have a position for this ticker

        levels = get_levels_at_bar(df, i)
        zones = find_confluence_zones(levels, confluence_pct)

        current_price = float(bar["Close"])
        prev_price = float(df.iloc[i - 1]["Close"]) if i > 0 else current_price

        atr = compute_atr(df.iloc[: i + 1], 14)
        if np.isnan(atr) or atr <= 0:
            continue

        for zone in zones:
            if zone["score"] < min_score:
                continue

            zp = zone["price"]
            lo = zone["band_lo"]
            hi = zone["band_hi"]

            setup_type: str | None = None
            direction: str | None = None

            # --- BREAK detection ---
            broke_up = prev_price <= hi and current_price > hi
            broke_dn = prev_price >= lo and current_price < lo

            if broke_up:
                setup_type, direction = "BREAK_LONG", "LONG"
            elif broke_dn:
                setup_type, direction = "BREAK_SHORT", "SHORT"

            # --- BOUNCE detection (only when --all-setups) ---
            if not breaks_only and setup_type is None:
                prox = atr * 0.5
                if current_price > hi and abs(current_price - zp) <= prox:
                    setup_type, direction = "BOUNCE_LONG", "LONG"
                elif current_price < lo and abs(current_price - zp) <= prox:
                    setup_type, direction = "BOUNCE_SHORT", "SHORT"

            if setup_type is None or direction is None:
                continue

            # Entry on next bar open
            entry_price = float(next_bar["Open"])
            risk_per_share = atr_stop_mult * atr

            if direction == "LONG":
                stop = entry_price - risk_per_share
                target = entry_price + rr * risk_per_share
            else:
                stop = entry_price + risk_per_share
                target = entry_price - rr * risk_per_share

            risk_dollars = equity * risk_pct
            shares = risk_dollars / risk_per_share if risk_per_share > 0 else 0
            if shares <= 0:
                continue

            position = Position(
                ticker=ticker,
                direction=direction,
                entry_date=next_date,
                entry_price=entry_price,
                stop=stop,
                target=target,
                shares=shares,
                risk_dollars=risk_dollars,
                setup_type=setup_type,
                zone_price=zp,
                zone_score=zone["score"],
            )
            break  # one trade per bar per ticker

    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    trades: list[dict[str, Any]],
    initial_equity: float,
) -> dict[str, Any]:
    """Aggregate performance metrics from a trade list."""
    if not trades:
        return {"error": "No trades executed"}

    df = pd.DataFrame(trades)

    total_trades = len(df)
    wins = df[df["win"]].shape[0]
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    gross_profit = df.loc[df["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = df.loc[df["win"], "pnl"].mean() if wins > 0 else 0.0
    avg_loss = df.loc[~df["win"], "pnl"].mean() if losses > 0 else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    net_pnl = df["pnl"].sum()
    final_equity = initial_equity + net_pnl
    total_return = net_pnl / initial_equity

    # Max drawdown (equity curve)
    df_sorted = df.sort_values("exit_date")
    equity_curve = initial_equity + df_sorted["pnl"].cumsum()
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())

    avg_hold = df["hold_bars"].mean()

    # By setup type
    setup_breakdown: dict[str, Any] = {}
    for st in df["setup_type"].unique():
        sub = df[df["setup_type"] == st]
        sub_wins = sub[sub["win"]].shape[0]
        setup_breakdown[st] = {
            "trades": len(sub),
            "win_rate": round(sub_wins / len(sub), 4),
            "net_pnl": round(sub["pnl"].sum(), 2),
        }

    # By ticker
    ticker_breakdown: dict[str, Any] = {}
    for tkr in df["ticker"].unique():
        sub = df[df["ticker"] == tkr]
        sub_wins = sub[sub["win"]].shape[0]
        ticker_breakdown[tkr] = {
            "trades": len(sub),
            "win_rate": round(sub_wins / len(sub), 4),
            "net_pnl": round(sub["pnl"].sum(), 2),
            "profitable": sub["pnl"].sum() > 0,
        }

    all_ticker_profitable = all(v["profitable"] for v in ticker_breakdown.values())

    # Monthly P&L
    df["exit_month"] = pd.to_datetime(df["exit_date"]).dt.to_period("M").astype(str)
    monthly_pnl = df.groupby("exit_month")["pnl"].sum()
    all_months_profitable = bool((monthly_pnl > 0).all())

    return {
        "initial_equity": round(initial_equity, 2),
        "final_equity": round(final_equity, 2),
        "net_pnl": round(net_pnl, 2),
        "total_return_pct": round(total_return * 100, 2),
        "total_trades": total_trades,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 4),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 4),
        "avg_hold_bars": round(avg_hold, 2),
        "all_tickers_profitable": all_ticker_profitable,
        "all_months_profitable": all_months_profitable,
        "setup_breakdown": setup_breakdown,
        "ticker_breakdown": ticker_breakdown,
        "monthly_pnl": monthly_pnl.round(2).to_dict(),
    }


# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict[str, Any]) -> None:
    """Print a formatted backtest summary to stdout."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  MEAN LEVELS BACKTEST – RESULTS SUMMARY")
    print(sep)

    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return

    print(f"  Equity:      ${metrics['initial_equity']:>12,.2f}  →  "
          f"${metrics['final_equity']:>12,.2f}")
    print(f"  Net P&L:     ${metrics['net_pnl']:>12,.2f}  "
          f"(+{metrics['total_return_pct']:.1f}%)")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.4f}%")
    print()
    print(f"  Total Trades: {metrics['total_trades']:>6}")
    print(f"  Wins/Losses:  {metrics['wins']} / {metrics['losses']}")
    print(f"  Win Rate:     {metrics['win_rate_pct']:.1f}%")
    print(f"  Profit Factor:{metrics['profit_factor']:.2f}")
    print(f"  Expectancy:  ${metrics['expectancy']:.2f}/trade")
    print(f"  Avg Win:     ${metrics['avg_win']:.2f}    "
          f"Avg Loss: ${metrics['avg_loss']:.2f}")
    print(f"  Avg Hold:     {metrics['avg_hold_bars']:.1f} bars")
    print()
    print(f"  All tickers profitable: {metrics['all_tickers_profitable']}")
    print(f"  All months profitable:  {metrics['all_months_profitable']}")

    print("\n  SETUP BREAKDOWN:")
    print(f"  {'Setup':<16} {'Trades':>6}  {'Win%':>6}  {'Net P&L':>12}")
    print("  " + "-" * 46)
    for st, v in metrics["setup_breakdown"].items():
        print(f"  {st:<16} {v['trades']:>6}  {v['win_rate']*100:>5.1f}%  "
              f"${v['net_pnl']:>10,.2f}")

    print("\n  TOP TICKERS BY P&L:")
    tkr_sorted = sorted(
        metrics["ticker_breakdown"].items(),
        key=lambda x: x[1]["net_pnl"], reverse=True,
    )
    print(f"  {'Ticker':<8} {'Trades':>6}  {'Win%':>6}  {'Net P&L':>12}")
    print("  " + "-" * 38)
    for tkr, v in tkr_sorted[:10]:
        flag = "✓" if v["profitable"] else "✗"
        print(f"  {tkr:<8} {v['trades']:>6}  {v['win_rate']*100:>5.1f}%  "
              f"${v['net_pnl']:>10,.2f}  {flag}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mean Levels Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Tickers to backtest (default: 20 from default list)",
    )
    parser.add_argument(
        "--months", type=int, default=DEFAULT_MONTHS, metavar="N",
        help=f"Months of history (default {DEFAULT_MONTHS})",
    )
    parser.add_argument(
        "--equity", type=float, default=DEFAULT_EQUITY, metavar="DOLLARS",
        help=f"Starting equity (default ${DEFAULT_EQUITY:,.0f})",
    )
    parser.add_argument(
        "--risk-pct", type=float, default=DEFAULT_RISK_PCT, metavar="FRAC",
        help=f"Risk per trade as fraction of equity (default {DEFAULT_RISK_PCT})",
    )
    parser.add_argument(
        "--atr-stop-mult", type=float, default=DEFAULT_ATR_STOP_MULT, metavar="X",
        help=f"ATR stop multiplier (default {DEFAULT_ATR_STOP_MULT})",
    )
    parser.add_argument(
        "--rr", type=float, default=DEFAULT_RR, metavar="RATIO",
        help=f"Reward-to-risk ratio for take-profit (default {DEFAULT_RR})",
    )
    parser.add_argument(
        "--breaks-only", action="store_true",
        help="Trade BREAK setups only (default)",
    )
    parser.add_argument(
        "--all-setups", action="store_true",
        help="Trade both BREAK and BOUNCE setups",
    )
    parser.add_argument(
        "--min-score", type=int, default=DEFAULT_MIN_SCORE, metavar="N",
        help=f"Minimum confluence score to trade (default {DEFAULT_MIN_SCORE})",
    )
    parser.add_argument(
        "--confluence-pct", type=float, default=DEFAULT_CONFLUENCE_PCT, metavar="PCT",
        help=f"Confluence band width (default {DEFAULT_CONFLUENCE_PCT})",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR, metavar="DIR",
        help=f"Output directory for results (default {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print JSON summary to stdout (no verbose output)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    tickers = args.tickers or DEFAULT_TICKERS[:20]
    breaks_only = not args.all_setups  # default to breaks-only unless --all-setups

    verbose = not args.json

    if verbose:
        print(f"Mean Levels Backtest  |  "
              f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"Tickers: {len(tickers)}  |  Months: {args.months}  |  "
              f"Equity: ${args.equity:,.0f}  |  Risk/trade: {args.risk_pct*100:.1f}%")
        mode = "BREAKS ONLY" if breaks_only else "BREAKS + BOUNCES"
        print(f"Mode: {mode}  |  Min score: {args.min_score}  |  R:R 1:{args.rr}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    all_trades: list[dict[str, Any]] = []
    failed: list[str] = []

    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"[{i:>2}/{len(tickers)}] Backtesting {ticker} ...", flush=True)
        try:
            df = fetch_history(ticker, args.months)
            trades = simulate_ticker(
                ticker=ticker,
                df=df,
                initial_equity=args.equity,
                risk_pct=args.risk_pct,
                atr_stop_mult=args.atr_stop_mult,
                rr=args.rr,
                breaks_only=breaks_only,
                min_score=args.min_score,
                confluence_pct=args.confluence_pct,
            )
            all_trades.extend(trades)
            if verbose:
                print(f"          {len(trades)} trades")
        except Exception as exc:
            failed.append(ticker)
            if verbose:
                print(f"          ERROR: {exc}")

    if verbose:
        print(f"\nTotal trades collected: {len(all_trades)}")
        if failed:
            print(f"Failed tickers: {', '.join(failed)}")

    metrics = compute_metrics(all_trades, args.equity)

    # Save trade log
    trades_path = os.path.join(args.output_dir, "trades.json")
    with open(trades_path, "w") as fh:
        json.dump(all_trades, fh, indent=2)

    # Save summary
    summary_payload = {
        "backtest_time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": {
            "tickers": tickers,
            "months": args.months,
            "equity": args.equity,
            "risk_pct": args.risk_pct,
            "atr_stop_mult": args.atr_stop_mult,
            "rr": args.rr,
            "breaks_only": breaks_only,
            "min_score": args.min_score,
            "confluence_pct": args.confluence_pct,
        },
        "metrics": metrics,
        "failed_tickers": failed,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary_payload, fh, indent=2)

    if args.json:
        print(json.dumps(summary_payload, separators=(",", ":")))
    else:
        print_metrics(metrics)
        print(f"Trade log  → {trades_path}")
        print(f"Summary    → {summary_path}")


if __name__ == "__main__":
    main()
