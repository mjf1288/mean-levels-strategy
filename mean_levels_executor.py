#!/usr/bin/env python3
"""
Mean Levels Live Executor
=========================
Reads scanner results and places live trades via Public.com API.

Design:
  - BREAKS ONLY (BREAK_LONG / BREAK_SELL) — bounces filtered per backtest v2
  - 1% risk per trade (configurable via --risk-pct)
  - Limit entry at zone center (retest of broken level)
  - Separate stop-loss order (STOP) placed after entry fills or immediately as GTC
  - Score >= 3 required; score 3 gets half-size, score >= 4 gets full size
  - Max concurrent positions capped to prevent overconcentration
  - Dry-run by default; pass --live to place real orders

Usage:
  python3 mean_levels_executor.py                    # Dry-run (default)
  python3 mean_levels_executor.py --live             # Place real orders
  python3 mean_levels_executor.py --risk-pct 2.0     # 2% risk per trade
  python3 mean_levels_executor.py --min-score 5      # Only high-conviction
  python3 mean_levels_executor.py --max-positions 5  # Cap at 5 orders
"""

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------------

def _ensure(package: str, import_name: str = None) -> None:
    name = import_name or package
    try:
        __import__(name)
    except ImportError:
        print(f"[bootstrap] Installing {package} …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "--quiet"],
            stdout=subprocess.DEVNULL,
        )

_ensure("publicdotcom-py", "public_api_sdk")

from public_api_sdk import (
    PublicApiClient,
    PublicApiClientConfiguration,
    OrderRequest,
    PreflightRequest,
    OrderInstrument,
    InstrumentType,
    OrderSide,
    OrderType,
    OrderExpirationRequest,
    TimeInForce,
)
from public_api_sdk.auth_config import ApiKeyAuthConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_PATH = "/home/user/workspace/mean_levels_results.json"
EXECUTION_LOG_PATH = "/home/user/workspace/mean_levels_execution_log.json"
ACCOUNT_ID = "5OF28683"

# ANSI
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

def green(s): return f"{_GREEN}{s}{_RESET}"
def red(s): return f"{_RED}{s}{_RESET}"
def yellow(s): return f"{_YELLOW}{s}{_RESET}"
def cyan(s): return f"{_CYAN}{s}{_RESET}"
def bold(s): return f"{_BOLD}{s}{_RESET}"

# ---------------------------------------------------------------------------
# Load scanner results
# ---------------------------------------------------------------------------

def load_scanner_results(path: str = RESULTS_PATH) -> List[Dict]:
    """Load and return scanner results from JSON file."""
    if not os.path.exists(path):
        print(red(f"ERROR: Scanner results not found at {path}"))
        print("Run the scanner first: python3 mean_levels_scanner.py")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    results = data.get("results", [])
    scan_time = data.get("scan_time", "unknown")
    print(f"[info] Loaded scanner results from {scan_time}")
    print(f"[info] {len(results)} tickers scanned")
    return results


# ---------------------------------------------------------------------------
# Filter to break-only setups
# ---------------------------------------------------------------------------

def filter_break_setups(
    results: List[Dict],
    min_score: int = 3,
    long_only: bool = True,
) -> List[Dict]:
    """
    Extract only BREAK_LONG (and optionally BREAK_SELL) setups with score >= min_score.
    Returns a flat list of dicts with ticker info attached.

    long_only=True (default) filters out all SELL setups because Public.com
    does not support short selling on equity accounts.
    """
    allowed_types = {"BREAK_LONG"}
    if not long_only:
        allowed_types.add("BREAK_SELL")

    setups = []
    skipped_sells = 0
    for r in results:
        if r is None:
            continue
        ticker = r["ticker"]
        price = r["price"]
        for setup in r.get("setups", []):
            setup_type = setup.get("type", "")
            score = setup.get("score", 0)

            # Filter: breaks only, min score
            if setup_type not in ("BREAK_LONG", "BREAK_SELL"):
                continue
            if score < min_score:
                continue
            # Long-only filter
            if setup_type not in allowed_types:
                skipped_sells += 1
                continue

            setups.append({
                "ticker": ticker,
                "price": price,
                "type": setup_type,
                "score": score,
                "entry": setup["entry"],
                "stop": setup["stop"],
                "t1": setup["t1"],
                "t2": setup["t2"],
                "t3": setup["t3"],
                "risk_per_share": setup["risk_per_share"],
                "levels": "+".join(setup["zone"]["levels"]),
            })

    if skipped_sells > 0 and long_only:
        print(f"[info] Skipped {skipped_sells} SELL setup(s) — long-only mode (Public.com)")

    # Sort by score descending (best setups first)
    setups.sort(key=lambda x: x["score"], reverse=True)
    return setups


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def size_position(
    equity: float,
    entry: float,
    stop: float,
    risk_pct: float,
    score: int,
) -> Dict:
    """
    Calculate position size using fixed-fractional risk model.

    Returns dict with: shares, risk_dollars, position_value
    """
    risk_per_share = abs(entry - stop)
    if risk_per_share == 0:
        return {"shares": 0, "risk_dollars": 0, "position_value": 0}

    risk_dollars = equity * (risk_pct / 100.0)

    # Half-size for score 3 (marginal confluence)
    if score <= 3:
        risk_dollars *= 0.5

    shares = int(risk_dollars / risk_per_share)
    if shares < 1:
        shares = 1

    position_value = round(shares * entry, 2)

    return {
        "shares": shares,
        "risk_dollars": round(risk_dollars, 2),
        "position_value": position_value,
        "risk_per_share": round(risk_per_share, 4),
    }


# ---------------------------------------------------------------------------
# Fetch live equity from Public.com
# ---------------------------------------------------------------------------

def get_live_equity(secret: str) -> float:
    """Fetch total equity from Public.com portfolio."""
    try:
        client = PublicApiClient(
            ApiKeyAuthConfig(api_secret_key=secret),
            config=PublicApiClientConfiguration(default_account_number=ACCOUNT_ID),
        )
        portfolio = client.get_portfolio()
        total_equity = sum(e.value for e in portfolio.equity)
        client.close()
        return float(total_equity)
    except Exception as e:
        print(yellow(f"[warn] Could not fetch live equity: {e}"))
        print(yellow("[warn] Falling back to --equity override or default"))
        return 0.0


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

def execute_setup(
    client: PublicApiClient,
    setup: Dict,
    sizing: Dict,
    dry_run: bool = True,
) -> Dict:
    """
    Place entry order (LIMIT) for a single setup.

    For BREAK_LONG: BUY at limit = entry price (zone center retest)
    For BREAK_SELL: SELL at limit = entry price

    Returns execution result dict.
    """
    ticker = setup["ticker"]
    entry = setup["entry"]
    stop = setup["stop"]
    shares = sizing["shares"]
    setup_type = setup["type"]

    side = OrderSide.BUY if "LONG" in setup_type else OrderSide.SELL
    side_str = "BUY" if "LONG" in setup_type else "SELL"

    result = {
        "ticker": ticker,
        "type": setup_type,
        "side": side_str,
        "shares": shares,
        "entry": entry,
        "stop": stop,
        "score": setup["score"],
        "levels": setup["levels"],
        "position_value": sizing["position_value"],
        "risk_dollars": sizing["risk_dollars"],
        "timestamp": datetime.now().isoformat(),
        "status": "DRY_RUN",
        "entry_order_id": None,
        "stop_order_id": None,
        "errors": [],
    }

    if dry_run:
        print(f"  [DRY RUN] {side_str} {shares} x {ticker} @ ${entry:.2f} "
              f"(stop ${stop:.2f}) — score {setup['score']} [{setup['levels']}]")
        print(f"            Risk: ${sizing['risk_dollars']:.2f} | "
              f"Position: ${sizing['position_value']:,.2f}")
        return result

    # --- LIVE EXECUTION ---

    # Step 1: Place entry order (LIMIT, GTC so it persists overnight if needed)
    try:
        entry_order_id = str(uuid.uuid4())
        entry_request = OrderRequest(
            order_id=entry_order_id,
            instrument=OrderInstrument(
                symbol=ticker,
                type=InstrumentType.EQUITY,
            ),
            order_side=side,
            order_type=OrderType.LIMIT,
            quantity=shares,
            limit_price=Decimal(str(round(entry, 2))),
            expiration=OrderExpirationRequest(time_in_force=TimeInForce.DAY),
        )
        entry_response = client.place_order(entry_request)
        result["entry_order_id"] = entry_response.order_id
        result["status"] = "ENTRY_PLACED"
        print(green(f"  ENTRY ORDER PLACED: {side_str} {shares} x {ticker} "
                     f"@ ${entry:.2f} — ID: {entry_response.order_id}"))
    except Exception as e:
        result["status"] = "ENTRY_FAILED"
        result["errors"].append(f"Entry order failed: {str(e)}")
        print(red(f"  ENTRY FAILED: {ticker} — {e}"))
        return result

    # Step 2: Place protective stop order
    # For BREAK_LONG (BUY): stop is a SELL STOP below entry
    # For BREAK_SELL (SELL): stop is a BUY STOP above entry
    try:
        stop_side = OrderSide.SELL if "LONG" in setup_type else OrderSide.BUY
        stop_order_id = str(uuid.uuid4())
        stop_request = OrderRequest(
            order_id=stop_order_id,
            instrument=OrderInstrument(
                symbol=ticker,
                type=InstrumentType.EQUITY,
            ),
            order_side=stop_side,
            order_type=OrderType.STOP,
            quantity=shares,
            stop_price=Decimal(str(round(stop, 2))),
            expiration=OrderExpirationRequest(time_in_force=TimeInForce.DAY),
        )
        stop_response = client.place_order(stop_request)
        result["stop_order_id"] = stop_response.order_id
        result["status"] = "ENTRY_AND_STOP_PLACED"
        print(green(f"  STOP ORDER PLACED: {'SELL' if 'LONG' in setup_type else 'BUY'} "
                     f"{shares} x {ticker} @ ${stop:.2f} — ID: {stop_response.order_id}"))
    except Exception as e:
        result["errors"].append(f"Stop order failed: {str(e)}")
        print(yellow(f"  [warn] Stop order failed for {ticker}: {e}"))
        print(yellow(f"  Entry is live without a stop — MANAGE MANUALLY"))

    return result


# ---------------------------------------------------------------------------
# Execution log
# ---------------------------------------------------------------------------

def save_execution_log(executions: List[Dict], path: str = EXECUTION_LOG_PATH):
    """Append execution results to the log file."""
    existing = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, KeyError):
            existing = []

    existing.extend(executions)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n[info] Execution log saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Place real orders (default is dry-run)",
    )
    parser.add_argument(
        "--risk-pct", type=float, default=1.0,
        help="Percent of equity to risk per trade (default: 1.0)",
    )
    parser.add_argument(
        "--min-score", type=int, default=3,
        help="Minimum confluence score to trade (default: 3)",
    )
    parser.add_argument(
        "--max-positions", type=int, default=8,
        help="Maximum number of orders to place in one session (default: 8)",
    )
    parser.add_argument(
        "--equity", type=float, default=None,
        help="Override account equity (default: fetched from Public.com)",
    )
    parser.add_argument(
        "--results-file", type=str, default=RESULTS_PATH,
        help="Path to scanner results JSON",
    )
    parser.add_argument(
        "--allow-shorts", action="store_true",
        help="Allow SELL setups (Public.com does not support shorts; off by default)",
    )
    args = parser.parse_args()

    # --- Header ---
    mode = red("LIVE") if args.live else yellow("DRY RUN")
    print()
    print(bold("=" * 70))
    print(bold(f"  MEAN LEVELS EXECUTOR — {mode}"))
    print(bold(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}"))
    print(bold(f"  Risk: {args.risk_pct}% | Min Score: {args.min_score} | "
               f"Max Positions: {args.max_positions}"))
    print(bold("=" * 70))

    # --- API Setup ---
    secret = os.environ.get("PUBLIC_COM_SECRET")
    if not secret:
        print(red("ERROR: PUBLIC_COM_SECRET not set"))
        sys.exit(1)

    # --- Get equity ---
    if args.equity:
        equity = args.equity
        print(f"\n[info] Using override equity: ${equity:,.2f}")
    else:
        equity = get_live_equity(secret)
        if equity <= 0:
            print(red("ERROR: Could not determine account equity. Use --equity to override."))
            sys.exit(1)
        print(f"\n[info] Live account equity: ${equity:,.2f}")

    risk_per_trade = equity * (args.risk_pct / 100.0)
    print(f"[info] Risk per trade ({args.risk_pct}%): ${risk_per_trade:,.2f}")

    # --- Load and filter setups ---
    results = load_scanner_results(args.results_file)
    long_only = not args.allow_shorts
    setups = filter_break_setups(results, min_score=args.min_score, long_only=long_only)

    if not setups:
        print(yellow("\nNo qualifying break setups found. Nothing to execute."))
        return

    print(f"\n[info] {len(setups)} break setup(s) found (score >= {args.min_score})")

    # Cap at max positions
    if len(setups) > args.max_positions:
        print(f"[info] Capping at {args.max_positions} (highest score first)")
        setups = setups[:args.max_positions]

    # --- Size and display ---
    print()
    print(bold(f"{'#':<3} {'TICKER':<7} {'TYPE':<12} {'SCORE':>5} {'LEVELS':<16} "
               f"{'ENTRY':>8} {'STOP':>8} {'SHARES':>6} {'VALUE':>10} {'RISK':>8}"))
    print("-" * 100)

    # Fetch buying power for cumulative cap
    try:
        bp_client = PublicApiClient(
            ApiKeyAuthConfig(api_secret_key=secret),
            config=PublicApiClientConfiguration(default_account_number=ACCOUNT_ID),
        )
        portfolio = bp_client.get_portfolio()
        buying_power = float(portfolio.buying_power.buying_power)
        bp_client.close()
        print(f"[info] Available buying power: ${buying_power:,.2f}")
    except Exception:
        buying_power = equity * 2.0  # conservative estimate with 2:1 margin
        print(f"[info] Estimated buying power (2:1 margin): ${buying_power:,.2f}")

    cumulative_notional = 0.0
    # Use 90% of buying power — reserve 10% as margin buffer
    usable_bp = buying_power * 0.90

    sized_setups = []
    for i, setup in enumerate(setups, 1):
        sizing = size_position(
            equity=equity,
            entry=setup["entry"],
            stop=setup["stop"],
            risk_pct=args.risk_pct,
            score=setup["score"],
        )

        if sizing["shares"] == 0:
            continue

        # Check if adding this position would exceed usable buying power
        if cumulative_notional + sizing["position_value"] > usable_bp:
            print(yellow(f"  [skip] {setup['ticker']} — cumulative notional "
                         f"(${cumulative_notional + sizing['position_value']:,.2f}) "
                         f"would exceed usable buying power (${usable_bp:,.2f})"))
            continue

        cumulative_notional += sizing["position_value"]
        sized_setups.append((setup, sizing))

        color = green if "LONG" in setup["type"] else red
        print(color(
            f"{i:<3} {setup['ticker']:<7} {setup['type']:<12} {setup['score']:>5} "
            f"{setup['levels']:<16} {setup['entry']:>8.2f} {setup['stop']:>8.2f} "
            f"{sizing['shares']:>6} ${sizing['position_value']:>9,.2f} "
            f"${sizing['risk_dollars']:>7,.2f}"
        ))

    if not sized_setups:
        print(yellow("\nNo setups passed sizing filters."))
        return

    total_value = sum(s["position_value"] for _, s in sized_setups)
    total_risk = sum(s["risk_dollars"] for _, s in sized_setups)
    print(f"\n{'':3} {'TOTAL':<7} {'':12} {'':>5} {'':16} {'':>8} {'':>8} "
          f"{'':>6} ${total_value:>9,.2f} ${total_risk:>7,.2f}")

    # --- Execute ---
    if not args.live:
        print(yellow(f"\n--- DRY RUN COMPLETE ---"))
        print(yellow(f"To place real orders, re-run with --live"))
        print()

        # Still save dry-run log
        executions = []
        for setup, sizing in sized_setups:
            executions.append(execute_setup(None, setup, sizing, dry_run=True))
        save_execution_log(executions)
        return

    # Live mode — create client and execute
    print(bold(f"\n>>> PLACING {len(sized_setups)} LIVE ORDER(S) <<<"))
    print()

    client = PublicApiClient(
        ApiKeyAuthConfig(api_secret_key=secret),
        config=PublicApiClientConfiguration(default_account_number=ACCOUNT_ID),
    )

    executions = []
    placed = 0
    for setup, sizing in sized_setups:
        result = execute_setup(client, setup, sizing, dry_run=False)
        executions.append(result)
        if "PLACED" in result["status"]:
            placed += 1
        print()

    client.close()

    print(bold(f"\nExecution complete: {placed}/{len(sized_setups)} orders placed"))

    # Save log
    save_execution_log(executions)


if __name__ == "__main__":
    main()
