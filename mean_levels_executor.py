"""
mean_levels_executor.py
=======================
Live / Paper Trade Executor for the Mean Levels Strategy
---------------------------------------------------------
Reads scanner output from mean_levels_results.json and converts active setups
into actionable order recommendations with full position sizing, stops, and
targets.  State is persisted to executor_state.json so it survives restarts.

Risk Management Defaults (v2 config – breaks-only, optimised)
--------------------------------------------------------------
  Risk per trade   : 2 % of current equity
  Stop-loss        : 1 × ATR below/above entry
  Take-profit      : 2 × risk (R:R = 1:2)
  Setup filter     : BREAK setups only
  Min zone score   : 5
  Max open positions: 3

Modes
-----
  --dry-run  (default)  Print recommendations without recording state changes.
  --live                Apply recommendations to live state (persist positions).
  --status              Show current open positions and daily P&L.

State File (executor_state.json)
---------------------------------
  Stores: open positions, closed trades log, running equity, daily stats.
  Created automatically on first run; never contains API keys or secrets.

Usage
-----
  python3 mean_levels_executor.py --dry-run
  python3 mean_levels_executor.py --live
  python3 mean_levels_executor.py --status
  python3 mean_levels_executor.py --close AAPL 182.50
  python3 mean_levels_executor.py --risk-pct 0.01 --min-score 6 --dry-run
"""

import argparse
import json
import os
import sys
from datetime import datetime, date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_SCANNER_OUTPUT: str = "mean_levels_results.json"
DEFAULT_STATE_FILE: str = "executor_state.json"
DEFAULT_EQUITY: float = 100_000.0
DEFAULT_RISK_PCT: float = 0.02          # 2 % of equity
DEFAULT_ATR_STOP_MULT: float = 1.0
DEFAULT_RR: float = 2.0
DEFAULT_MIN_SCORE: int = 5
DEFAULT_MAX_POSITIONS: int = 3
BREAKS_ONLY: bool = True               # default to break setups only


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _empty_state(equity: float) -> dict[str, Any]:
    return {
        "created": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "last_updated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "equity": equity,
        "initial_equity": equity,
        "open_positions": {},   # ticker → position dict
        "closed_trades": [],
        "daily_stats": {},      # date str → {pnl, trades}
        "session_count": 0,
    }


def load_state(state_file: str, initial_equity: float) -> dict[str, Any]:
    """Load state from disk or create a fresh state."""
    if os.path.exists(state_file):
        try:
            with open(state_file) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Could not load state ({exc}); starting fresh.")
    return _empty_state(initial_equity)


def save_state(state: dict[str, Any], state_file: str) -> None:
    """Persist state to disk (atomic write via temp file)."""
    state["last_updated"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    tmp = state_file + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, state_file)


# ---------------------------------------------------------------------------
# Fetching current price & ATR
# ---------------------------------------------------------------------------

def get_current_price_and_atr(ticker: str) -> tuple[float, float]:
    """Return (latest_close, 14-bar ATR) for *ticker*."""
    try:
        df = yf.download(ticker, period="1mo", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        price = float(df["Close"].iloc[-1])
        hi, lo, cl = df["High"], df["Low"], df["Close"].shift(1)
        tr = pd.concat([(hi - lo), (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        return price, atr
    except Exception as exc:
        raise RuntimeError(f"Could not fetch {ticker}: {exc}") from exc


# ---------------------------------------------------------------------------
# Order generation
# ---------------------------------------------------------------------------

def generate_order(
    ticker: str,
    setup_type: str,
    price: float,
    atr: float,
    equity: float,
    risk_pct: float,
    atr_stop_mult: float,
    rr: float,
) -> dict[str, Any]:
    """Build an order recommendation dict from a setup signal.

    Parameters
    ----------
    ticker : str
    setup_type : str  – 'BREAK_LONG', 'BREAK_SHORT', etc.
    price : float     – current market price (entry estimate)
    atr : float       – 14-bar ATR
    equity : float    – current account equity
    risk_pct : float  – fraction of equity to risk
    atr_stop_mult : float
    rr : float        – reward-to-risk ratio

    Returns
    -------
    Order dict with entry, stop, target, shares, risk_dollars.
    """
    direction = "LONG" if "LONG" in setup_type else "SHORT"
    risk_per_share = atr_stop_mult * atr
    risk_dollars = equity * risk_pct
    shares = risk_dollars / risk_per_share if risk_per_share > 0 else 0.0

    if direction == "LONG":
        stop = price - risk_per_share
        target = price + rr * risk_per_share
    else:
        stop = price + risk_per_share
        target = price - rr * risk_per_share

    return {
        "ticker": ticker,
        "direction": direction,
        "setup_type": setup_type,
        "entry_price": round(price, 4),
        "stop": round(stop, 4),
        "target": round(target, 4),
        "shares": round(shares, 2),
        "risk_dollars": round(risk_dollars, 2),
        "risk_pct": round(risk_pct * 100, 2),
        "atr": round(atr, 4),
        "rr": rr,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


# ---------------------------------------------------------------------------
# Scanner result processing
# ---------------------------------------------------------------------------

def load_scanner_results(scanner_file: str) -> list[dict[str, Any]]:
    """Load and return the per-ticker results from a scanner JSON file."""
    if not os.path.exists(scanner_file):
        raise FileNotFoundError(
            f"Scanner output not found: {scanner_file}\n"
            "Run mean_levels_scanner.py first."
        )
    with open(scanner_file) as fh:
        data = json.load(fh)
    return data.get("results", [])


def filter_signals(
    results: list[dict[str, Any]],
    breaks_only: bool,
    min_score: int,
    already_open: set[str],
    max_positions: int,
    current_open: int,
) -> list[dict[str, Any]]:
    """Filter scanner results down to actionable signals.

    Parameters
    ----------
    results : list
        Per-ticker scan results from scanner.
    breaks_only : bool
        If True, only BREAK_LONG and BREAK_SHORT setups pass.
    min_score : int
        Minimum zone confluence score.
    already_open : set[str]
        Tickers that already have an open position (skip them).
    max_positions : int
        Maximum total open positions allowed.
    current_open : int
        Number of positions currently open.

    Returns
    -------
    Filtered list of (result, setup) tuples as dicts.
    """
    signals: list[dict[str, Any]] = []
    slots_remaining = max_positions - current_open

    for result in results:
        if result.get("error"):
            continue
        ticker = result["ticker"]
        if ticker in already_open:
            continue

        for setup in result.get("setups", []):
            stype = setup["type"]
            if breaks_only and "BREAK" not in stype:
                continue
            if setup["zone_score"] < min_score:
                continue

            signals.append({
                "ticker": ticker,
                "price": result["price"],
                "atr": result.get("atr"),
                "setup": setup,
                "levels": result.get("levels", {}),
                "scan_time": result.get("scan_time"),
            })
            break  # one signal per ticker

        if len(signals) >= slots_remaining:
            break

    return signals


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

def open_position(
    state: dict[str, Any],
    order: dict[str, Any],
) -> None:
    """Record a new open position in state (--live mode)."""
    ticker = order["ticker"]
    state["open_positions"][ticker] = {
        **order,
        "open_date": date.today().isoformat(),
        "status": "OPEN",
    }


def close_position(
    state: dict[str, Any],
    ticker: str,
    exit_price: float,
) -> dict[str, Any] | None:
    """Close a position manually and record the trade."""
    pos = state["open_positions"].pop(ticker, None)
    if pos is None:
        return None

    direction = pos["direction"]
    entry = pos["entry_price"]
    shares = pos["shares"]

    raw_pnl = (exit_price - entry) * shares
    pnl = raw_pnl if direction == "LONG" else -raw_pnl

    state["equity"] += pnl

    trade = {
        **pos,
        "exit_price": round(exit_price, 4),
        "pnl": round(pnl, 2),
        "exit_date": date.today().isoformat(),
        "exit_reason": "MANUAL",
        "win": pnl > 0,
        "equity_after": round(state["equity"], 2),
    }
    state["closed_trades"].append(trade)
    return trade


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

SEP = "=" * 72


def print_status(state: dict[str, Any]) -> None:
    """Print current account status."""
    print(f"\n{SEP}")
    print("  MEAN LEVELS EXECUTOR – STATUS")
    print(SEP)
    equity = state["equity"]
    initial = state["initial_equity"]
    net_pnl = equity - initial
    ret_pct = net_pnl / initial * 100

    print(f"  Equity  : ${equity:>12,.2f}")
    print(f"  Initial : ${initial:>12,.2f}")
    print(f"  Net P&L : ${net_pnl:>+12,.2f}  ({ret_pct:+.2f}%)")
    print(f"  Sessions: {state['session_count']}")
    print(f"  Updated : {state['last_updated']}")

    open_pos = state["open_positions"]
    print(f"\n  OPEN POSITIONS ({len(open_pos)}):")
    if open_pos:
        print(f"  {'Ticker':<8} {'Dir':<6} {'Entry':>8} {'Stop':>8} "
              f"{'Target':>8} {'Shares':>8} {'Risk $':>8}")
        print("  " + "-" * 60)
        for tkr, pos in open_pos.items():
            print(f"  {tkr:<8} {pos['direction']:<6} "
                  f"{pos['entry_price']:>8.2f} {pos['stop']:>8.2f} "
                  f"{pos['target']:>8.2f} {pos['shares']:>8.1f} "
                  f"${pos['risk_dollars']:>7.2f}")
    else:
        print("  (none)")

    closed = state["closed_trades"]
    print(f"\n  RECENT CLOSED TRADES (last 5 of {len(closed)}):")
    if closed:
        recent = closed[-5:]
        for t in reversed(recent):
            flag = "WIN " if t.get("win") else "LOSS"
            print(f"  [{flag}] {t['ticker']:<6} {t['direction']:<5} "
                  f"entry={t['entry_price']:.2f} exit={t['exit_price']:.2f} "
                  f"P&L=${t['pnl']:+.2f}  {t['exit_date']}")
    else:
        print("  (none)")
    print(SEP + "\n")


def print_orders(
    orders: list[dict[str, Any]],
    dry_run: bool = True,
) -> None:
    """Print generated order recommendations."""
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"\n{SEP}")
    print(f"  MEAN LEVELS EXECUTOR – ORDER RECOMMENDATIONS  [{mode}]")
    print(SEP)

    if not orders:
        print("  No actionable signals found.\n")
        print(SEP + "\n")
        return

    for i, order in enumerate(orders, 1):
        direction_str = "▲ LONG " if order["direction"] == "LONG" else "▼ SHORT"
        print(f"\n  [{i}] {order['ticker']}  {direction_str}  ({order['setup_type']})")
        print(f"      Entry  : ${order['entry_price']:.2f}")
        print(f"      Stop   : ${order['stop']:.2f}  "
              f"(−{abs(order['entry_price']-order['stop']):.2f}, 1×ATR)")
        print(f"      Target : ${order['target']:.2f}  "
              f"(+{abs(order['target']-order['entry_price']):.2f}, {order['rr']}R)")
        print(f"      Shares : {order['shares']:.1f}  "
              f"Risk: ${order['risk_dollars']:.2f} ({order['risk_pct']:.1f}%)")

    print(f"\n  Total signals: {len(orders)}")
    print(SEP + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mean Levels Live/Paper Trade Executor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Generate order recommendations without changing state (default)",
    )
    mode_group.add_argument(
        "--live", action="store_true",
        help="Apply recommendations and persist positions to state file",
    )
    mode_group.add_argument(
        "--status", action="store_true",
        help="Show current open positions and account stats",
    )

    parser.add_argument(
        "--close", nargs=2, metavar=("TICKER", "PRICE"),
        help="Manually close a position: --close AAPL 182.50",
    )
    parser.add_argument(
        "--scanner-file", default=DEFAULT_SCANNER_OUTPUT, metavar="FILE",
        help=f"Scanner output to read (default: {DEFAULT_SCANNER_OUTPUT})",
    )
    parser.add_argument(
        "--state-file", default=DEFAULT_STATE_FILE, metavar="FILE",
        help=f"State persistence file (default: {DEFAULT_STATE_FILE})",
    )
    parser.add_argument(
        "--equity", type=float, default=DEFAULT_EQUITY, metavar="DOLLARS",
        help=f"Initial equity when creating fresh state (default ${DEFAULT_EQUITY:,.0f})",
    )
    parser.add_argument(
        "--risk-pct", type=float, default=DEFAULT_RISK_PCT, metavar="FRAC",
        help=f"Risk per trade as fraction of equity (default {DEFAULT_RISK_PCT})",
    )
    parser.add_argument(
        "--min-score", type=int, default=DEFAULT_MIN_SCORE, metavar="N",
        help=f"Minimum confluence zone score (default {DEFAULT_MIN_SCORE})",
    )
    parser.add_argument(
        "--max-positions", type=int, default=DEFAULT_MAX_POSITIONS, metavar="N",
        help=f"Maximum concurrent open positions (default {DEFAULT_MAX_POSITIONS})",
    )
    parser.add_argument(
        "--all-setups", action="store_true",
        help="Include BOUNCE setups in addition to BREAKs",
    )
    parser.add_argument(
        "--atr-stop-mult", type=float, default=DEFAULT_ATR_STOP_MULT, metavar="X",
        help=f"ATR stop multiplier (default {DEFAULT_ATR_STOP_MULT})",
    )
    parser.add_argument(
        "--rr", type=float, default=DEFAULT_RR, metavar="RATIO",
        help=f"Reward-to-risk ratio (default {DEFAULT_RR})",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of formatted table",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    live_mode = args.live
    status_mode = args.status

    state = load_state(args.state_file, args.equity)

    # --- Manual close ---
    if args.close:
        ticker_arg, price_arg = args.close
        try:
            exit_p = float(price_arg)
        except ValueError:
            print(f"ERROR: Invalid price '{price_arg}'")
            sys.exit(1)
        trade = close_position(state, ticker_arg.upper(), exit_p)
        if trade is None:
            print(f"No open position found for {ticker_arg.upper()}")
        else:
            print(f"Closed {trade['ticker']}: P&L = ${trade['pnl']:+.2f}")
            save_state(state, args.state_file)
        return

    # --- Status mode ---
    if status_mode:
        print_status(state)
        return

    # --- Load scanner results ---
    try:
        scan_results = load_scanner_results(args.scanner_file)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    scan_time = None
    if os.path.exists(args.scanner_file):
        with open(args.scanner_file) as fh:
            raw = json.load(fh)
            scan_time = raw.get("scan_time", "unknown")

    print(f"\nMean Levels Executor  |  "
          f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Scanner data: {scan_time}  |  "
          f"Equity: ${state['equity']:,.2f}")

    breaks_only = not args.all_setups
    already_open = set(state["open_positions"].keys())
    current_open = len(already_open)

    signals = filter_signals(
        results=scan_results,
        breaks_only=breaks_only,
        min_score=args.min_score,
        already_open=already_open,
        max_positions=args.max_positions,
        current_open=current_open,
    )

    orders: list[dict[str, Any]] = []

    for sig in signals:
        ticker = sig["ticker"]
        price = sig.get("price")
        atr = sig.get("atr")

        # Refresh live price/ATR if data is stale or missing
        if price is None or atr is None:
            try:
                price, atr = get_current_price_and_atr(ticker)
            except RuntimeError as exc:
                print(f"  [WARN] Skipping {ticker}: {exc}")
                continue

        order = generate_order(
            ticker=ticker,
            setup_type=sig["setup"]["type"],
            price=price,
            atr=atr,
            equity=state["equity"],
            risk_pct=args.risk_pct,
            atr_stop_mult=args.atr_stop_mult,
            rr=args.rr,
        )
        order["zone_price"] = sig["setup"]["zone_price"]
        order["zone_score"] = sig["setup"]["zone_score"]
        order["zone_levels"] = sig["setup"]["zone_levels"]
        orders.append(order)

    if args.json:
        payload = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "mode": "live" if live_mode else "dry_run",
            "equity": state["equity"],
            "open_positions": list(state["open_positions"].keys()),
            "orders": orders,
        }
        print(json.dumps(payload, separators=(",", ":")))
    else:
        print_orders(orders, dry_run=not live_mode)

    # Apply to state in --live mode
    if live_mode and orders:
        for order in orders:
            open_position(state, order)
        state["session_count"] += 1
        save_state(state, args.state_file)
        print(f"State saved to {args.state_file}")
        print(f"Opened {len(orders)} position(s).")
    elif not live_mode and orders:
        print("(dry-run: state not modified)")


if __name__ == "__main__":
    main()
