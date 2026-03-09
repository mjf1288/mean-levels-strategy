"""
mean_levels_universe.py
=======================
Dynamic Universe Selector for the Mean Levels Strategy
-------------------------------------------------------
Ranks ~110 candidate equities by a composite score of:

  score = short_term_volume_ratio * ATR * relative_strength

Then filters by:
  - Minimum average daily volume  (default 1,000,000 shares)
  - Minimum ATR                   (default $1.00)

And selects the top N names (default 30) to pass to the scanner.

Candidate pool
--------------
  • S&P 500 heavyweights (mega-cap tech, financials, healthcare, energy)
  • Nasdaq-100 high-beta names
  • High-beta cyclicals (semiconductors, EV, biotech)
  • Sector ETFs (SPY, QQQ, IWM, XLF, XLE, XLK, XLV, SMH, ARKK, GLD, SLV, USO)

Usage
-----
  python3 mean_levels_universe.py               # print ranked universe
  python3 mean_levels_universe.py --run-scanner  # select then pipe to scanner
  python3 mean_levels_universe.py --top 40       # select top 40 instead of 30
  python3 mean_levels_universe.py --min-atr 2.0  # filter for high-vol names only
  python3 mean_levels_universe.py --min-vol 2000000
  python3 mean_levels_universe.py --json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Candidate pool (~110 tickers)
# ---------------------------------------------------------------------------
CANDIDATE_POOL: list[str] = [
    # S&P 500 mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
    "AVGO", "ORCL", "CRM", "ADBE", "AMD", "QCOM", "TXN", "INTC",
    "MU", "KLAC", "LRCX", "AMAT",
    # Nasdaq-100 high-beta
    "NFLX", "PANW", "CRWD", "SNOW", "DDOG", "TEAM", "ZS", "NET",
    "MRVL", "SMCI", "ARM", "PLTR", "RBLX", "UBER", "LYFT", "ABNB",
    "HOOD", "COIN", "MSTR", "SHOP",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "V", "MA",
    "PYPL", "SQ", "AXP",
    # Healthcare / Biotech
    "UNH", "LLY", "JNJ", "PFE", "ABBV", "MRK", "AMGN", "GILD",
    "REGN", "BIIB", "MRNA",
    # Energy
    "XOM", "CVX", "COP", "OXY", "SLB", "HAL",
    # Consumer
    "COST", "WMT", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD",
    # Industrials / Aerospace
    "CAT", "DE", "LMT", "RTX", "BA", "GE", "HON",
    # China tech ADRs
    "BABA", "JD", "BIDU", "NIO", "LI", "XPEV",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA",
    "XLF", "XLE", "XLK", "XLV", "XLC", "XLY", "XLI",
    "SMH", "SOXX", "ARKK",
    "GLD", "SLV", "USO", "TLT", "HYG",
]

# Deduplicate while preserving order
_seen: set[str] = set()
CANDIDATE_POOL = [t for t in CANDIDATE_POOL if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TOP_N: int = 30
DEFAULT_MIN_VOL: int = 1_000_000   # shares/day
DEFAULT_MIN_ATR: float = 1.0       # dollars
RS_BENCHMARK: str = "SPY"          # relative strength benchmark
RS_PERIOD_DAYS: int = 20           # look-back for relative strength
VOL_SHORT_DAYS: int = 5            # short-term volume ratio window


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _download_batch(tickers: list[str], period: str = "3mo") -> pd.DataFrame:
    """Bulk-download daily Close prices for multiple tickers."""
    try:
        raw = yf.download(
            tickers, period=period, interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
        )
    except Exception as exc:
        raise RuntimeError(f"Batch download failed: {exc}") from exc
    return raw


def compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return a per-ticker ATR Series from a multi-ticker OHLCV DataFrame."""
    atrs: dict[str, float] = {}
    if isinstance(df.columns, pd.MultiIndex):
        tickers_in_df = df.columns.get_level_values(0).unique().tolist()
        for tkr in tickers_in_df:
            try:
                sub = df[tkr][["High", "Low", "Close"]].dropna()
                if len(sub) < period + 1:
                    continue
                hi, lo, cl = sub["High"], sub["Low"], sub["Close"].shift(1)
                tr = pd.concat([(hi - lo), (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
                atrs[tkr] = float(tr.rolling(period).mean().iloc[-1])
            except Exception:
                pass
    return pd.Series(atrs)


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def rank_universe(
    min_vol: int = DEFAULT_MIN_VOL,
    min_atr: float = DEFAULT_MIN_ATR,
    top_n: int = DEFAULT_TOP_N,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download data and rank candidate tickers by composite score.

    Returns a DataFrame with columns:
      ticker, price, atr, avg_vol, vol_ratio, rs_20d, composite_score, rank
    """
    if verbose:
        print(f"Downloading data for {len(CANDIDATE_POOL)} candidates ...", flush=True)

    # We need individual downloads for accurate OHLCV
    rows: list[dict[str, Any]] = []

    for i, ticker in enumerate(CANDIDATE_POOL, 1):
        try:
            df = yf.download(
                ticker, period="3mo", interval="1d",
                auto_adjust=True, progress=False,
            )
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

            if len(df) < 20:
                continue

            price = float(df["Close"].iloc[-1])

            # ATR
            hi, lo, cl = df["High"], df["Low"], df["Close"].shift(1)
            tr = pd.concat([(hi - lo), (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])

            # Volume stats
            avg_vol = float(df["Volume"].tail(20).mean())
            short_vol = float(df["Volume"].tail(VOL_SHORT_DAYS).mean())
            vol_ratio = short_vol / avg_vol if avg_vol > 0 else 0.0

            # Relative strength vs SPY (price % change)
            ret_ticker = (
                (df["Close"].iloc[-1] - df["Close"].iloc[-RS_PERIOD_DAYS - 1])
                / df["Close"].iloc[-RS_PERIOD_DAYS - 1]
            )

            rows.append({
                "ticker": ticker,
                "price": round(price, 2),
                "atr": round(atr, 3),
                "avg_vol": int(avg_vol),
                "vol_ratio": round(vol_ratio, 4),
                "ret_20d": round(float(ret_ticker), 6),
            })

            if verbose and i % 20 == 0:
                print(f"  Processed {i}/{len(CANDIDATE_POOL)} ...", flush=True)

        except Exception:
            pass  # Skip silently

    df_ranks = pd.DataFrame(rows)
    if df_ranks.empty:
        raise RuntimeError("No data could be downloaded from the candidate pool.")

    # --- Fetch SPY return for relative strength ---
    try:
        spy = yf.download(RS_BENCHMARK, period="3mo", interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy_ret = float(
            (spy["Close"].iloc[-1] - spy["Close"].iloc[-RS_PERIOD_DAYS - 1])
            / spy["Close"].iloc[-RS_PERIOD_DAYS - 1]
        )
    except Exception:
        spy_ret = 0.0

    df_ranks["rs_20d"] = df_ranks["ret_20d"] - spy_ret

    # --- Apply filters ---
    df_ranks = df_ranks[df_ranks["avg_vol"] >= min_vol].copy()
    df_ranks = df_ranks[df_ranks["atr"] >= min_atr].copy()

    if df_ranks.empty:
        raise RuntimeError(
            f"No tickers passed filters: min_vol={min_vol}, min_atr={min_atr}"
        )

    # --- Normalise components to [0, 1] ---
    def norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0.0

    df_ranks["norm_vol_ratio"] = norm(df_ranks["vol_ratio"])
    df_ranks["norm_atr"] = norm(df_ranks["atr"])
    df_ranks["norm_rs"] = norm(df_ranks["rs_20d"])

    # Composite score (equal-weight of the three normalised components)
    df_ranks["composite_score"] = (
        df_ranks["norm_vol_ratio"]
        + df_ranks["norm_atr"]
        + df_ranks["norm_rs"]
    ) / 3.0

    df_ranks.sort_values("composite_score", ascending=False, inplace=True)
    df_ranks.reset_index(drop=True, inplace=True)
    df_ranks["rank"] = df_ranks.index + 1

    return df_ranks.head(top_n)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_ranked_table(df: pd.DataFrame) -> None:
    """Pretty-print the ranked universe table."""
    sep = "=" * 90
    print(f"\n{sep}")
    print("  MEAN LEVELS UNIVERSE SELECTOR – RANKED RESULTS")
    print(sep)
    print(f"  {'#':>3}  {'TICKER':<7}  {'PRICE':>8}  {'ATR':>6}  "
          f"{'AVG VOL':>12}  {'VOL RATIO':>9}  {'RS 20D':>8}  {'SCORE':>7}")
    print("-" * 90)
    for _, row in df.iterrows():
        vol_m = row["avg_vol"] / 1_000_000
        print(
            f"  {row['rank']:>3}  {row['ticker']:<7}  {row['price']:>8.2f}  "
            f"{row['atr']:>6.2f}  {vol_m:>10.1f}M  {row['vol_ratio']:>9.3f}  "
            f"{row['rs_20d']:>8.4f}  {row['composite_score']:>7.4f}"
        )
    print(sep)
    print(f"\n  Selected {len(df)} tickers\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mean Levels Universe Selector – pick best tickers to scan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--top", type=int, default=DEFAULT_TOP_N, metavar="N",
        help=f"Number of tickers to select (default {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--min-vol", type=int, default=DEFAULT_MIN_VOL, metavar="SHARES",
        help=f"Minimum avg daily volume filter (default {DEFAULT_MIN_VOL:,})",
    )
    parser.add_argument(
        "--min-atr", type=float, default=DEFAULT_MIN_ATR, metavar="DOLLARS",
        help=f"Minimum ATR filter in dollars (default {DEFAULT_MIN_ATR})",
    )
    parser.add_argument(
        "--run-scanner", action="store_true",
        help="After selection, pipe chosen tickers into mean_levels_scanner.py",
    )
    parser.add_argument(
        "--output", default="universe_selected.json", metavar="FILE",
        help="Output JSON file path (default: universe_selected.json)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Suppress progress output; print compact JSON to stdout",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    verbose = not args.json

    if verbose:
        print(f"Mean Levels Universe Selector  |  "
              f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"Candidates: {len(CANDIDATE_POOL)}  |  "
              f"Target top: {args.top}  |  "
              f"Min vol: {args.min_vol:,}  |  "
              f"Min ATR: ${args.min_atr:.2f}\n")

    ranked = rank_universe(
        min_vol=args.min_vol,
        min_atr=args.min_atr,
        top_n=args.top,
        verbose=verbose,
    )

    selected_tickers = ranked["ticker"].tolist()

    output_payload = {
        "selection_time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "top_n": args.top,
        "min_vol": args.min_vol,
        "min_atr": args.min_atr,
        "candidate_count": len(CANDIDATE_POOL),
        "selected_count": len(selected_tickers),
        "tickers": selected_tickers,
        "ranked_data": ranked.to_dict(orient="records"),
    }

    if args.json:
        print(json.dumps(output_payload, separators=(",", ":")))
    else:
        print_ranked_table(ranked)

    with open(args.output, "w") as fh:
        json.dump(output_payload, fh, indent=2)

    if verbose:
        print(f"Universe saved to {args.output}")
        print(f"Selected: {' '.join(selected_tickers)}\n")

    if args.run_scanner:
        if verbose:
            print("Launching mean_levels_scanner.py with selected universe ...\n")
        cmd = [
            sys.executable, "mean_levels_scanner.py",
            "--tickers", *selected_tickers,
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
