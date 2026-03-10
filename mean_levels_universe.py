#!/usr/bin/env python3
"""
Mean Levels Universe Selector
==============================
Dynamically selects the best ~30 tickers for the Mean Levels S/R Scanner
based on liquidity, volatility, and recent momentum.

Criteria:
  1. Liquidity  — 20-day average dollar volume ≥ $100M/day
  2. Movement   — 5-day ATR% (Average True Range / Price) ≥ 1.0%
  3. Not parabolics — 5-day return between -15% and +15% (filters blow-off moves)
  4. Price       — ≥ $5 (avoids penny stock noise)

Sources from S&P 500 + Nasdaq-100 + popular high-volume names.

Usage:
  python3 mean_levels_universe.py                  # Select top 30, print and save
  python3 mean_levels_universe.py --top 40          # Select top 40
  python3 mean_levels_universe.py --json            # JSON output
  python3 mean_levels_universe.py --run-scanner     # Select universe, then run scanner

Output:
  Saves the selected tickers to /home/user/workspace/mean_levels_watchlist.txt
  (one ticker per line) for use by mean_levels_scanner.py
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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

_ensure("yfinance")
_ensure("pandas")

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def green(s):  return f"{_GREEN}{s}{_RESET}"
def red(s):    return f"{_RED}{s}{_RESET}"
def yellow(s): return f"{_YELLOW}{s}{_RESET}"
def cyan(s):   return f"{_CYAN}{s}{_RESET}"
def bold(s):   return f"{_BOLD}{s}{_RESET}"

# ---------------------------------------------------------------------------
# Ticker universe — broad pool to screen from
# ---------------------------------------------------------------------------

# S&P 500 mega/large caps + Nasdaq-100 heavyweights + popular high-beta names
# ~120 names that cover the most-traded US equities
CANDIDATE_POOL = [
    # Broad market ETFs (always include)
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "ARKK",
    # Mega cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "NFLX", "CRM",
    # Semiconductors
    "AMD", "INTC", "MU", "MRVL", "QCOM", "ARM", "SMCI", "TSM", "AMAT", "LRCX",
    # Software / Cloud
    "PLTR", "SNOW", "DDOG", "CRWD", "NET", "PANW", "ZS", "SHOP", "UBER", "DASH",
    # Financials
    "JPM", "GS", "MS", "BAC", "C", "WFC", "BLK", "SCHW", "COF", "AXP",
    # Consumer / Retail
    "COST", "WMT", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD", "CMG", "LULU",
    # Healthcare / Pharma
    "UNH", "LLY", "JNJ", "PFE", "ABBV", "MRK", "BMY", "AMGN", "GILD", "ISRG",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY", "DVN", "MPC", "PSX", "VLO", "HAL",
    # Industrials / Aerospace
    "BA", "CAT", "DE", "GE", "HON", "LMT", "RTX", "UPS", "FDX", "MMM",
    # Payment networks
    "V", "MA", "PYPL",
    # Crypto-adjacent / High-beta
    "COIN", "MARA", "MSTR", "RIOT", "HOOD",
    # Telecom / Media
    "DIS", "CMCSA", "T", "VZ",
    # Other popular large caps
    "ORCL", "IBM", "ADBE", "NOW", "INTU", "TXN", "KO", "PEP", "PG", "ABT",
]

# De-duplicate
CANDIDATE_POOL = list(dict.fromkeys(CANDIDATE_POOL))

# ---------------------------------------------------------------------------
# Screening logic
# ---------------------------------------------------------------------------

def screen_universe(
    candidates: List[str],
    top_n: int = 30,
    min_avg_dollar_vol: float = 100_000_000,   # $100M/day
    min_atr_pct: float = 1.0,                   # 1.0% daily ATR
    max_5d_return_abs: float = 15.0,            # filter parabolics
    min_price: float = 5.0,
    quiet: bool = False,
) -> List[Dict]:
    """
    Screen candidates and return the top_n ranked by a composite score
    of liquidity + movement.

    Composite score = log(avg_dollar_volume) * atr_pct
    This favours stocks that are BOTH liquid AND moving.
    """
    import math

    if not quiet:
        print(cyan(f"\nScreening {len(candidates)} candidates …"))

    # Download 25 days of data in one batch for speed
    try:
        raw = yf.download(
            candidates,
            period="25d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as exc:
        print(red(f"ERROR downloading data: {exc}"))
        return []

    results = []

    for ticker in candidates:
        try:
            # Extract this ticker's data from the multi-ticker download
            if len(candidates) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()

            df.dropna(subset=["Close", "Volume", "High", "Low"], inplace=True)

            if len(df) < 10:
                continue

            # Current price
            price = float(df["Close"].iloc[-1])
            if price < min_price:
                continue

            # 20-day average dollar volume
            recent_20 = df.iloc[-20:] if len(df) >= 20 else df
            avg_volume = float(recent_20["Volume"].mean())
            avg_dollar_vol = avg_volume * price

            if avg_dollar_vol < min_avg_dollar_vol:
                continue

            # 5-day ATR%  (Average True Range as % of price)
            recent_5 = df.iloc[-5:]
            tr_values = []
            for i in range(len(recent_5)):
                h = float(recent_5["High"].iloc[i])
                l = float(recent_5["Low"].iloc[i])
                if i == 0 and len(df) > 5:
                    prev_c = float(df["Close"].iloc[-6])
                elif i > 0:
                    prev_c = float(recent_5["Close"].iloc[i - 1])
                else:
                    prev_c = float(recent_5["Close"].iloc[i])
                tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                tr_values.append(tr)

            atr = sum(tr_values) / len(tr_values)
            atr_pct = (atr / price) * 100

            if atr_pct < min_atr_pct:
                continue

            # 5-day return (filter parabolics)
            close_5d_ago = float(df["Close"].iloc[-6]) if len(df) >= 6 else float(df["Close"].iloc[0])
            return_5d = ((price - close_5d_ago) / close_5d_ago) * 100

            if abs(return_5d) > max_5d_return_abs:
                continue

            # Composite score: liquidity × movement
            # log(dollar_vol) weights liquidity on a compressed scale
            # atr_pct weights how much the stock is actually moving
            composite = math.log10(avg_dollar_vol) * atr_pct

            results.append({
                "ticker":         ticker,
                "price":          round(price, 2),
                "avg_dollar_vol": round(avg_dollar_vol),
                "avg_dollar_vol_str": f"${avg_dollar_vol / 1e9:.1f}B" if avg_dollar_vol >= 1e9 else f"${avg_dollar_vol / 1e6:.0f}M",
                "avg_volume":     round(avg_volume),
                "atr_pct":        round(atr_pct, 2),
                "return_5d":      round(return_5d, 2),
                "composite":      round(composite, 2),
            })

        except Exception:
            continue

    # Sort by composite score (highest = most liquid + most movement)
    results.sort(key=lambda x: x["composite"], reverse=True)

    if not quiet:
        print(f"  {len(results)} passed filters out of {len(candidates)}")

    return results[:top_n]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

WATCHLIST_PATH = "/home/user/workspace/mean_levels_watchlist.txt"
UNIVERSE_JSON  = "/home/user/workspace/mean_levels_universe.json"


def print_results(selected: List[Dict]) -> None:
    print()
    print(bold("=" * 95))
    print(bold("  MEAN LEVELS UNIVERSE — TOP PICKS"))
    print(bold("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")))
    print(bold("=" * 95))
    print()
    print(bold(
        f"{'#':>3}  {'TICKER':<7}  {'PRICE':>9}  {'AVG $ VOL':>10}  "
        f"{'ATR%':>6}  {'5D RET%':>8}  {'SCORE':>7}  {'WHY'}"
    ))
    print("-" * 95)

    for i, r in enumerate(selected, 1):
        # Tag the reason
        tags = []
        if r["atr_pct"] >= 3.0:
            tags.append("HIGH VOL")
        elif r["atr_pct"] >= 2.0:
            tags.append("VOLATILE")
        if r["avg_dollar_vol"] >= 5e9:
            tags.append("MEGA LIQ")
        elif r["avg_dollar_vol"] >= 1e9:
            tags.append("HIGH LIQ")
        if abs(r["return_5d"]) >= 5:
            tags.append("BIG MOVE")

        tag_str = ", ".join(tags) if tags else "SOLID"

        colour = green if r["return_5d"] >= 0 else red
        ret_str = colour(f"{r['return_5d']:>+7.2f}%")

        print(
            f"{i:>3}  {r['ticker']:<7}  "
            f"${r['price']:>8.2f}  "
            f"{r['avg_dollar_vol_str']:>10}  "
            f"{r['atr_pct']:>5.2f}%  "
            f"{ret_str}  "
            f"{r['composite']:>7.2f}  "
            f"{tag_str}"
        )

    print()


def save_watchlist(selected: List[Dict], path: str = WATCHLIST_PATH) -> None:
    tickers = [r["ticker"] for r in selected]
    with open(path, "w") as f:
        f.write("\n".join(tickers) + "\n")
    print(f"Watchlist saved → {path} ({len(tickers)} tickers)")


def save_json(selected: List[Dict], path: str = UNIVERSE_JSON) -> None:
    payload = {
        "scan_time": datetime.now().isoformat(),
        "count": len(selected),
        "tickers": selected,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of tickers to select (default: 30)",
    )
    parser.add_argument(
        "--min-dollar-vol", type=float, default=100_000_000,
        help="Minimum 20-day avg dollar volume (default: $100M)",
    )
    parser.add_argument(
        "--min-atr", type=float, default=1.0,
        help="Minimum 5-day ATR%% (default: 1.0%%)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print JSON output instead of table",
    )
    parser.add_argument(
        "--run-scanner", action="store_true",
        help="After selecting universe, immediately run mean_levels_scanner.py with the result",
    )
    parser.add_argument(
        "--equity", type=float, default=None,
        help="Account equity to pass to the scanner (only used with --run-scanner)",
    )

    args = parser.parse_args()

    selected = screen_universe(
        candidates=CANDIDATE_POOL,
        top_n=args.top,
        min_avg_dollar_vol=args.min_dollar_vol,
        min_atr_pct=args.min_atr,
        quiet=args.json,
    )

    if not selected:
        print(yellow("No tickers passed the screening filters."))
        return

    if args.json:
        print(json.dumps({"scan_time": datetime.now().isoformat(), "tickers": selected}, indent=2))
    else:
        print_results(selected)

    save_watchlist(selected)
    save_json(selected)

    # Optionally chain into the scanner
    if args.run_scanner:
        tickers = [r["ticker"] for r in selected]
        cmd = [
            sys.executable, "/home/user/workspace/mean_levels_scanner.py",
            "--tickers", *tickers,
        ]
        if args.equity:
            cmd.extend(["--equity", str(args.equity)])
        print(bold(f"\n>>> Running scanner on {len(tickers)} tickers …\n"))
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
