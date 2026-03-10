#!/usr/bin/env python3
"""
Mean Levels — Consolidated Live System
=======================================
One script. One cron. 9 AM ET.

Flow:
  1. Cancel all stale open orders
  2. Compute regime: DSS Bressert + Lyapunov HP → BUY / SELL / NEUTRAL
  3. Compute mean levels: CDM, PDM, CMM, PMM (Lua-faithful engine)
  4. If regime = BUY → place limit BUY orders at all 4 mean levels
     If regime = SELL → skip (Public.com can't short)
     If regime = NEUTRAL → skip (no edge)
  5. Each order gets a protective STOP placed alongside
  6. Log everything to JSON

Risk parameters:
  - 1% equity per trade
  - Max 3 limit orders per ticker
  - Min confluence score: N/A (orders at all 4 levels individually)
  - Long-only (Public.com constraint)
  - DAY time-in-force (orders expire at close if unfilled)

Usage:
  python3 mean_levels_live.py                          # Dry-run (default)
  python3 mean_levels_live.py --live                   # Place real orders
  python3 mean_levels_live.py --live --tickers SPY QQQ # Specific tickers
  python3 mean_levels_live.py --live --risk-pct 1.5    # Override risk %
"""

import argparse
import json
import math
import os
import subprocess
import sys
import uuid
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _ensure(pkg, name=None):
    try:
        __import__(name or pkg)
    except ImportError:
        print(f"[bootstrap] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                              stdout=subprocess.DEVNULL)

_ensure("yfinance")
_ensure("pandas")
_ensure("publicdotcom-py", "public_api_sdk")

import pandas as pd
import yfinance as yf
import importlib.util as _ilu

from public_api_sdk import (
    PublicApiClient, PublicApiClientConfiguration,
    OrderRequest, PreflightRequest, OrderInstrument,
    InstrumentType, OrderSide, OrderType,
    OrderExpirationRequest, TimeInForce,
)
from public_api_sdk.auth_config import ApiKeyAuthConfig

# ---------------------------------------------------------------------------
# Import indicator modules
# ---------------------------------------------------------------------------

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

_DSS = DSSBressert()   # stoch=13, smooth=8, signal=8
_LYAP = LyapunovHP()   # filter=7, L_Period=525

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCOUNT_ID = "5OF28683"
LOG_PATH = "/home/user/workspace/mean_levels_live_log.json"
RESULTS_PATH = "/home/user/workspace/mean_levels_results.json"

# ANSI
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; B = "\033[1m"; X = "\033[0m"
def green(s): return f"{G}{s}{X}"
def red(s): return f"{R}{s}{X}"
def yellow(s): return f"{Y}{s}{X}"
def cyan(s): return f"{C}{s}{X}"
def bold(s): return f"{B}{s}{X}"

LEVEL_WEIGHTS = {"PMM": 4, "CMM": 3, "PDM": 2, "CDM": 1}

DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "TSLA", "AMD", "NFLX", "CRM", "AVGO", "JPM", "V",
    "MA", "UNH", "LLY", "COST",
]

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, days: int = 800) -> Optional[pd.DataFrame]:
    """Download daily OHLCV (800 days for Lyapunov warmup)."""
    try:
        raw = yf.download(ticker, period=f"{days}d", interval="1d",
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        raw.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        if hasattr(raw.index, "tz") and raw.index.tz is not None:
            raw.index = raw.index.tz_convert("America/New_York").tz_localize(None)
        raw.sort_index(inplace=True)
        return raw if len(raw) >= 30 else None
    except Exception as e:
        print(yellow(f"  [warn] {ticker}: fetch failed — {e}"))
        return None

# ---------------------------------------------------------------------------
# Mean levels (Lua-faithful)
# ---------------------------------------------------------------------------

def _calc_mean_engine(closes, groups):
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
    }


def compute_mean_levels(df: pd.DataFrame) -> Dict[str, float]:
    closes = df["Close"].tolist()
    daily_groups = [(ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()) for ts in df.index]
    daily = _calc_mean_engine(closes, daily_groups)
    monthly_groups = [(ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()).strftime("%Y-%m") for ts in df.index]
    monthly = _calc_mean_engine(closes, monthly_groups)
    return {
        "CDM": daily["current"],
        "PDM": daily["prev"],
        "CMM": monthly["current"],
        "PMM": monthly["prev"],
    }

# ---------------------------------------------------------------------------
# Regime computation (DSS + Lyapunov)
# ---------------------------------------------------------------------------

def compute_regime(df: pd.DataFrame) -> Dict:
    """
    Returns dict with:
      regime: 'BUY' | 'SELL' | 'NEUTRAL'
      dss_direction, lyap_direction, dss_score, lyap_score, dss_value, lyap_value
    """
    dss_result = _DSS.calculate(df)
    dss_score = _DSS.get_signal_score(df)
    lyap_result = _LYAP.calculate_from_df(df)
    lyap_score = _LYAP.get_signal_score(df)

    dss_dir = dss_result["direction"] if dss_result else "NEUTRAL"
    lyap_dir = lyap_result["direction"] if lyap_result else "NEUTRAL"

    # Net direction
    if dss_dir == "BULLISH" and lyap_dir == "BULLISH":
        net = "BULLISH"
    elif dss_dir == "BEARISH" and lyap_dir == "BEARISH":
        net = "BEARISH"
    elif dss_dir == "NEUTRAL" and lyap_dir != "NEUTRAL":
        net = lyap_dir
    elif lyap_dir == "NEUTRAL" and dss_dir != "NEUTRAL":
        net = dss_dir
    else:
        net = "NEUTRAL"

    # Map to regime
    if net == "BULLISH":
        regime = "BUY"
    elif net == "BEARISH":
        regime = "SELL"
    else:
        regime = "NEUTRAL"

    return {
        "regime": regime,
        "net_direction": net,
        "dss_direction": dss_dir,
        "dss_score": dss_score,
        "dss_value": dss_result["dss"] if dss_result else None,
        "lyap_direction": lyap_dir,
        "lyap_score": lyap_score,
        "lyap_value": lyap_result["lyapunov"] if lyap_result else None,
    }

# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def size_position(equity: float, entry: float, stop: float,
                  risk_pct: float = 1.0, weight: int = 1) -> Dict:
    risk_per_share = abs(entry - stop)
    if risk_per_share == 0:
        return {"shares": 0, "risk_dollars": 0, "position_value": 0}
    risk_dollars = equity * (risk_pct / 100.0)
    # Scale by level weight (PMM=4x, CMM=3x, PDM=2x, CDM=1x → normalize)
    # Higher weight = stronger level = full size. Lower weight = smaller.
    weight_factor = min(weight / 4.0, 1.0)  # PMM=1.0, CMM=0.75, PDM=0.5, CDM=0.25
    risk_dollars *= max(weight_factor, 0.25)
    shares = int(risk_dollars / risk_per_share)
    if shares < 1:
        shares = 1
    return {
        "shares": shares,
        "risk_dollars": round(risk_dollars, 2),
        "position_value": round(shares * entry, 2),
    }

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def get_client(secret: str) -> PublicApiClient:
    return PublicApiClient(
        ApiKeyAuthConfig(api_secret_key=secret),
        config=PublicApiClientConfiguration(default_account_number=ACCOUNT_ID),
    )

# ---------------------------------------------------------------------------
# Step 1: Cancel all stale orders
# ---------------------------------------------------------------------------

def cancel_all_orders(client: PublicApiClient, dry_run: bool = True) -> List[str]:
    """Cancel all open orders. Returns list of cancelled order IDs."""
    cancelled = []
    try:
        portfolio = client.get_portfolio()
        if not portfolio.orders:
            print("  No open orders to cancel.")
            return cancelled

        for order in portfolio.orders:
            status = order.status.value if hasattr(order.status, 'value') else str(order.status)
            if status in ("FILLED", "CANCELLED", "REJECTED", "EXPIRED"):
                continue
            oid = order.order_id
            sym = order.instrument.symbol
            side = order.side.value if hasattr(order.side, 'value') else str(order.side)
            otype = order.type.value if hasattr(order.type, 'value') else str(order.type)

            if dry_run:
                print(f"  [DRY] Would cancel: {oid[:8]}... {side} {sym} ({otype})")
            else:
                try:
                    client.cancel_order(order_id=oid, account_id=ACCOUNT_ID)
                    print(f"  Cancelled: {oid[:8]}... {side} {sym} ({otype})")
                    cancelled.append(oid)
                except Exception as e:
                    print(yellow(f"  [warn] Failed to cancel {oid[:8]}: {e}"))
    except Exception as e:
        print(yellow(f"  [warn] Could not fetch orders: {e}"))

    return cancelled

# ---------------------------------------------------------------------------
# Step 4: Place limit orders at mean levels
# ---------------------------------------------------------------------------

def place_level_orders(
    client: PublicApiClient,
    ticker: str,
    price: float,
    levels: Dict[str, float],
    equity: float,
    risk_pct: float,
    max_orders: int,
    dry_run: bool = True,
) -> List[Dict]:
    """
    Place limit BUY orders at each mean level that is BELOW current price
    (buy the dip to the level). Each order gets a stop.

    Returns list of execution records.
    """
    executions = []

    # Sort levels by weight (strongest first): PMM > CMM > PDM > CDM
    level_items = sorted(levels.items(), key=lambda x: LEVEL_WEIGHTS.get(x[0], 0), reverse=True)

    orders_placed = 0
    for level_name, level_price in level_items:
        if orders_placed >= max_orders:
            break

        # Only place BUY limit orders at levels BELOW current price
        # (we're buying pullbacks to support)
        if level_price >= price:
            print(f"    {level_name} = ${level_price:.2f} — above price, skip")
            continue

        # How far below?  Skip if level is >5% below price (too far)
        distance_pct = (price - level_price) / price * 100
        if distance_pct > 5.0:
            print(f"    {level_name} = ${level_price:.2f} — {distance_pct:.1f}% below, too far, skip")
            continue

        # Stop = 0.7% below the level (same as BREAK_LONG)
        stop_price = round(level_price * 0.993, 2)
        weight = LEVEL_WEIGHTS.get(level_name, 1)

        sizing = size_position(equity, level_price, stop_price, risk_pct, weight)
        if sizing["shares"] == 0:
            continue

        entry_price = round(level_price, 2)
        shares = sizing["shares"]

        record = {
            "ticker": ticker,
            "level": level_name,
            "level_price": entry_price,
            "stop_price": stop_price,
            "shares": shares,
            "risk_dollars": sizing["risk_dollars"],
            "position_value": sizing["position_value"],
            "weight": weight,
            "distance_pct": round(distance_pct, 2),
            "status": "DRY_RUN",
            "entry_order_id": None,
            "stop_order_id": None,
            "errors": [],
            "timestamp": datetime.now().isoformat(),
        }

        if dry_run:
            print(green(
                f"    [DRY] BUY {shares} x {ticker} @ ${entry_price:.2f} "
                f"({level_name}, wt={weight}) stop=${stop_price:.2f} "
                f"risk=${sizing['risk_dollars']:.2f}"
            ))
            executions.append(record)
            orders_placed += 1
            continue

        # --- LIVE: Place entry limit order ---
        try:
            entry_oid = str(uuid.uuid4())
            entry_req = OrderRequest(
                order_id=entry_oid,
                instrument=OrderInstrument(symbol=ticker, type=InstrumentType.EQUITY),
                order_side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=shares,
                limit_price=Decimal(str(entry_price)),
                expiration=OrderExpirationRequest(time_in_force=TimeInForce.DAY),
            )
            resp = client.place_order(entry_req)
            record["entry_order_id"] = resp.order_id
            record["status"] = "ENTRY_PLACED"
            print(green(
                f"    ENTRY: BUY {shares} x {ticker} @ ${entry_price:.2f} "
                f"({level_name}) — ID: {resp.order_id[:8]}"
            ))
        except Exception as e:
            record["status"] = "ENTRY_FAILED"
            record["errors"].append(str(e))
            print(red(f"    ENTRY FAILED: {ticker} {level_name} — {e}"))
            executions.append(record)
            continue

        # --- LIVE: Place protective stop ---
        try:
            stop_oid = str(uuid.uuid4())
            stop_req = OrderRequest(
                order_id=stop_oid,
                instrument=OrderInstrument(symbol=ticker, type=InstrumentType.EQUITY),
                order_side=OrderSide.SELL,
                order_type=OrderType.STOP,
                quantity=shares,
                stop_price=Decimal(str(stop_price)),
                expiration=OrderExpirationRequest(time_in_force=TimeInForce.DAY),
            )
            stop_resp = client.place_order(stop_req)
            record["stop_order_id"] = stop_resp.order_id
            record["status"] = "ENTRY_AND_STOP_PLACED"
            print(green(f"    STOP:  SELL {shares} x {ticker} @ ${stop_price:.2f} — ID: {stop_resp.order_id[:8]}"))
        except Exception as e:
            record["errors"].append(f"Stop failed: {e}")
            print(yellow(f"    [warn] Stop failed for {ticker}: {e}"))

        executions.append(record)
        orders_placed += 1

    return executions


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def save_log(session: Dict, path: str = LOG_PATH):
    existing = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, KeyError):
            existing = []
    existing.append(session)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Place real orders")
    parser.add_argument("--tickers", nargs="+", metavar="T", help="Tickers to scan")
    parser.add_argument("--risk-pct", type=float, default=1.0, help="Risk %% per trade (default: 1.0)")
    parser.add_argument("--max-orders", type=int, default=3, help="Max orders per ticker (default: 3)")
    parser.add_argument("--equity", type=float, default=None, help="Override equity")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS
    dry_run = not args.live
    mode_str = red("LIVE") if args.live else yellow("DRY RUN")

    print()
    print(bold("=" * 80))
    print(bold(f"  MEAN LEVELS LIVE — {mode_str}"))
    print(bold(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}"))
    print(bold(f"  Risk: {args.risk_pct}% | Max orders/ticker: {args.max_orders} | Long-only"))
    print(bold("=" * 80))

    # --- API setup ---
    secret = os.environ.get("PUBLIC_COM_SECRET")
    if not secret:
        print(red("ERROR: PUBLIC_COM_SECRET not set"))
        sys.exit(1)

    client = get_client(secret)

    # --- Get equity ---
    if args.equity:
        equity = args.equity
    else:
        try:
            portfolio = client.get_portfolio()
            equity = float(sum(e.value for e in portfolio.equity))
            buying_power = float(portfolio.buying_power.buying_power)
        except Exception as e:
            print(red(f"ERROR: Could not fetch equity: {e}"))
            sys.exit(1)

    print(f"\n[info] Account equity: ${equity:,.2f}")

    # --- Step 1: Cancel all stale orders ---
    print(bold("\n── STEP 1: Cancel stale orders ──"))
    cancelled = cancel_all_orders(client, dry_run=dry_run)
    print(f"  Cancelled: {len(cancelled)} order(s)")

    # --- Step 2+3+4: For each ticker, compute regime → levels → place orders ---
    print(bold(f"\n── STEP 2: Scan {len(tickers)} ticker(s) ──"))

    session = {
        "timestamp": datetime.now().isoformat(),
        "mode": "LIVE" if args.live else "DRY_RUN",
        "equity": equity,
        "risk_pct": args.risk_pct,
        "cancelled_orders": cancelled,
        "tickers": {},
    }

    total_orders = 0
    total_buy = 0
    total_skip_regime = 0
    all_results = []

    for i, ticker in enumerate(tickers, 1):
        ticker = ticker.upper()
        print(f"\n  [{i:>2}/{len(tickers)}] {bold(ticker)}")

        # Fetch data
        df = fetch_ohlcv(ticker)
        if df is None:
            print(yellow("    No data, skip"))
            continue

        price = round(float(df.iloc[-1]["Close"]), 2)
        print(f"    Price: ${price:.2f}")

        # Compute regime
        regime_info = compute_regime(df)
        regime = regime_info["regime"]
        dss_d = regime_info["dss_direction"][:4]
        lyap_d = regime_info["lyap_direction"][:4]

        if regime == "BUY":
            regime_str = green(f"BUY  (DSS={dss_d} Lyap={lyap_d})")
        elif regime == "SELL":
            regime_str = red(f"SELL (DSS={dss_d} Lyap={lyap_d})")
        else:
            regime_str = yellow(f"NEUTRAL (DSS={dss_d} Lyap={lyap_d})")
        print(f"    Regime: {regime_str}")

        # Compute mean levels
        levels = compute_mean_levels(df)
        print(f"    CDM=${levels['CDM']:.2f}  PDM=${levels['PDM']:.2f}  "
              f"CMM=${levels['CMM']:.2f}  PMM=${levels['PMM']:.2f}")

        # Save to results
        ticker_result = {
            "ticker": ticker,
            "price": price,
            "levels": {k: round(v, 4) for k, v in levels.items()},
            "regime": regime_info,
        }

        # Gate: only place orders if regime = BUY
        if regime != "BUY":
            print(yellow(f"    → Regime is {regime}, no orders"))
            total_skip_regime += 1
            ticker_result["orders"] = []
            ticker_result["action"] = "SKIPPED"
            all_results.append(ticker_result)
            session["tickers"][ticker] = ticker_result
            continue

        # Place limit orders at all 4 levels
        total_buy += 1
        executions = place_level_orders(
            client=client,
            ticker=ticker,
            price=price,
            levels=levels,
            equity=equity,
            risk_pct=args.risk_pct,
            max_orders=args.max_orders,
            dry_run=dry_run,
        )
        ticker_result["orders"] = executions
        ticker_result["action"] = "ORDERS_PLACED" if executions else "NO_LEVELS_BELOW"
        total_orders += len(executions)
        all_results.append(ticker_result)
        session["tickers"][ticker] = ticker_result

    client.close()

    # --- Summary ---
    print()
    print(bold("=" * 80))
    print(bold("  SESSION SUMMARY"))
    print(bold("=" * 80))
    print(f"  Tickers scanned:    {len(tickers)}")
    print(f"  Regime = BUY:       {total_buy}")
    print(f"  Regime = SKIP:      {total_skip_regime}")
    print(f"  Orders placed:      {total_orders}")
    print(f"  Stale cancelled:    {len(cancelled)}")
    print(bold("=" * 80))

    # Save
    session["summary"] = {
        "tickers_scanned": len(tickers),
        "regime_buy": total_buy,
        "regime_skip": total_skip_regime,
        "orders_placed": total_orders,
        "stale_cancelled": len(cancelled),
    }
    save_log(session)
    print(f"\n[info] Log saved → {LOG_PATH}")

    # Also save results for compatibility with old scanner format
    results_payload = {
        "scan_time": datetime.now().isoformat(),
        "results": all_results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_payload, f, indent=2, default=str)
    print(f"[info] Results saved → {RESULTS_PATH}")

    if args.json:
        print(json.dumps(session, indent=2, default=str))


if __name__ == "__main__":
    main()
