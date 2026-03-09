"""
mean_levels_scanner.py
======================
Mean Levels Support/Resistance Scanner
---------------------------------------
Computes the 4 MeanTF indicator levels for a list of equity tickers:

  CDM  – Current Day Mean      (mean of current/last trading day's OHLC)
  PDM  – Previous Day Mean     (mean of previous trading day's OHLC)
  CMM  – Current Month Mean    (mean of daily closes so far in calendar month)
  PMM  – Previous Month Mean   (mean of all daily closes in the prior calendar month)

Identifies confluence zones where 2+ levels stack within 0.5 % of each other,
scores those zones by weight (PMM=4, CMM=3, PDM=2, CDM=1), and detects four
setup types:

  BREAK_LONG   – price broke *above* a confluence zone
  BREAK_SHORT  – price broke *below* a confluence zone
  BOUNCE_LONG  – price is pulling back to a zone from above (long entry)
  BOUNCE_SHORT – price is pulling back to a zone from below (short entry)

Usage
-----
  python3 mean_levels_scanner.py
  python3 mean_levels_scanner.py --tickers AAPL MSFT NVDA
  python3 mean_levels_scanner.py --tickers AAPL --confluence-pct 0.75
  python3 mean_levels_scanner.py --json          # compact JSON output only
  python3 mean_levels_scanner.py --output results.json
"""

import argparse
import json
import sys
from datetime import datetime, date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Default universe – 30 high-liquidity names
# ---------------------------------------------------------------------------
DEFAULT_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AVGO", "AMD", "ORCL",
    "NFLX", "CRM", "ADBE", "QCOM", "TXN",
    "JPM", "BAC", "GS", "MS", "V",
    "UNH", "LLY", "JNJ", "PFE", "ABBV",
    "XOM", "CVX", "SPY", "QQQ", "IWM",
]

# Level weights for confluence scoring
LEVEL_WEIGHTS: dict[str, int] = {
    "PMM": 4,
    "CMM": 3,
    "PDM": 2,
    "CDM": 1,
}

# Default confluence band (fraction of price)
DEFAULT_CONFLUENCE_PCT: float = 0.005   # 0.5 %

# How close price must be to a zone to qualify as a BOUNCE setup (fraction)
BOUNCE_PROXIMITY_PCT: float = 0.010    # 1.0 %


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_ticker_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Download daily OHLCV data for *ticker* via yfinance.

    Returns a DataFrame with columns [Open, High, Low, Close, Volume].
    Raises ValueError if no data is returned.
    """
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
    except Exception as exc:
        raise ValueError(f"yfinance download failed for {ticker}: {exc}") from exc

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten multi-level columns that yfinance sometimes produces
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.dropna(subset=["Close"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Mean level computation
# ---------------------------------------------------------------------------

def compute_mean_levels(df: pd.DataFrame) -> dict[str, float | None]:
    """Compute CDM, PDM, CMM, PMM from a daily OHLC DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLCV data sorted ascending by date.

    Returns
    -------
    dict with keys 'CDM', 'PDM', 'CMM', 'PMM' (float or None if insufficient data).
    """
    today = datetime.utcnow().date()
    current_month = today.month
    current_year = today.year

    # Determine previous month boundaries
    if current_month == 1:
        prev_month, prev_year = 12, current_year - 1
    else:
        prev_month, prev_year = current_month - 1, current_year

    # --- CDM: mean(O, H, L, C) of the most recent trading day ---
    last_row = df.iloc[-1]
    cdm = float(np.mean([last_row["Open"], last_row["High"],
                          last_row["Low"], last_row["Close"]]))

    # --- PDM: mean(O, H, L, C) of the day before the most recent ---
    pdm: float | None = None
    if len(df) >= 2:
        prev_row = df.iloc[-2]
        pdm = float(np.mean([prev_row["Open"], prev_row["High"],
                              prev_row["Low"], prev_row["Close"]]))

    # --- CMM: mean of daily *closes* in the current calendar month ---
    mask_cm = (df.index.month == current_month) & (df.index.year == current_year)
    cmm_series = df.loc[mask_cm, "Close"]
    cmm: float | None = float(cmm_series.mean()) if not cmm_series.empty else None

    # --- PMM: mean of daily *closes* in the previous calendar month ---
    mask_pm = (df.index.month == prev_month) & (df.index.year == prev_year)
    pmm_series = df.loc[mask_pm, "Close"]
    pmm: float | None = float(pmm_series.mean()) if not pmm_series.empty else None

    return {"CDM": cdm, "PDM": pdm, "CMM": cmm, "PMM": pmm}


# ---------------------------------------------------------------------------
# Confluence zone detection
# ---------------------------------------------------------------------------

def find_confluence_zones(
    levels: dict[str, float | None],
    confluence_pct: float = DEFAULT_CONFLUENCE_PCT,
) -> list[dict[str, Any]]:
    """Group nearby mean levels into confluence zones.

    Two levels are considered confluent if they are within *confluence_pct*
    of their mid-price of each other.

    Parameters
    ----------
    levels : dict
        Keys are level names ('CDM', 'PDM', 'CMM', 'PMM'); values are prices.
    confluence_pct : float
        Half-band around each level expressed as a fraction of price.

    Returns
    -------
    List of zone dicts, each containing:
        - 'price'   : float  – average price of constituent levels
        - 'levels'  : list   – level names in this zone
        - 'score'   : int    – sum of weights for constituent levels
        - 'band_lo' : float  – lower edge of zone
        - 'band_hi' : float  – upper edge of zone
    """
    valid: list[tuple[str, float]] = [
        (name, val) for name, val in levels.items() if val is not None
    ]
    valid.sort(key=lambda x: x[1])

    zones: list[dict[str, Any]] = []
    used: set[str] = set()

    for i, (name_i, price_i) in enumerate(valid):
        if name_i in used:
            continue
        group_names = [name_i]
        group_prices = [price_i]

        for j, (name_j, price_j) in enumerate(valid):
            if i == j or name_j in used:
                continue
            mid = (price_i + price_j) / 2.0
            if abs(price_i - price_j) / mid <= confluence_pct:
                group_names.append(name_j)
                group_prices.append(price_j)

        if len(group_names) >= 2:
            avg_price = float(np.mean(group_prices))
            score = sum(LEVEL_WEIGHTS.get(n, 0) for n in group_names)
            band_lo = avg_price * (1 - confluence_pct)
            band_hi = avg_price * (1 + confluence_pct)
            zones.append({
                "price": round(avg_price, 4),
                "levels": sorted(group_names),
                "score": score,
                "band_lo": round(band_lo, 4),
                "band_hi": round(band_hi, 4),
            })
            used.update(group_names)

    # Deduplicate overlapping zones (keep highest score)
    zones.sort(key=lambda z: z["score"], reverse=True)
    deduplicated: list[dict[str, Any]] = []
    for zone in zones:
        overlap = False
        for existing in deduplicated:
            if zone["band_lo"] <= existing["band_hi"] and zone["band_hi"] >= existing["band_lo"]:
                overlap = True
                break
        if not overlap:
            deduplicated.append(zone)

    return deduplicated


# ---------------------------------------------------------------------------
# Setup detection
# ---------------------------------------------------------------------------

def detect_setups(
    current_price: float,
    prev_price: float,
    zones: list[dict[str, Any]],
    proximity_pct: float = BOUNCE_PROXIMITY_PCT,
) -> list[dict[str, Any]]:
    """Detect BREAK and BOUNCE setups relative to confluence zones.

    A BREAK is confirmed when price *crossed* the zone between the previous
    close and the current close.  A BOUNCE is signalled when price is within
    *proximity_pct* of the zone without having broken through.

    Parameters
    ----------
    current_price : float
        Most recent closing price.
    prev_price : float
        Closing price of the preceding bar (used to detect breaks).
    zones : list
        Output of :func:`find_confluence_zones`.
    proximity_pct : float
        Fraction of price defining the bounce proximity band.

    Returns
    -------
    List of setup dicts with keys: 'type', 'zone_price', 'zone_score',
    'zone_levels', 'proximity_pct_away'.
    """
    setups: list[dict[str, Any]] = []

    for zone in zones:
        zp = zone["price"]
        lo = zone["band_lo"]
        hi = zone["band_hi"]
        score = zone["score"]
        lvls = zone["levels"]

        pct_away = abs(current_price - zp) / zp

        # --- BREAK detection ---
        crossed_up = prev_price <= hi and current_price > hi
        crossed_dn = prev_price >= lo and current_price < lo

        if crossed_up:
            setups.append({
                "type": "BREAK_LONG",
                "zone_price": zp,
                "zone_score": score,
                "zone_levels": lvls,
                "proximity_pct_away": round(pct_away * 100, 3),
            })
        elif crossed_dn:
            setups.append({
                "type": "BREAK_SHORT",
                "zone_price": zp,
                "zone_score": score,
                "zone_levels": lvls,
                "proximity_pct_away": round(pct_away * 100, 3),
            })
        # --- BOUNCE detection (no break occurred) ---
        elif not crossed_up and not crossed_dn:
            if current_price > hi and pct_away <= proximity_pct:
                # Price is above zone and approaching from above → BOUNCE_LONG
                setups.append({
                    "type": "BOUNCE_LONG",
                    "zone_price": zp,
                    "zone_score": score,
                    "zone_levels": lvls,
                    "proximity_pct_away": round(pct_away * 100, 3),
                })
            elif current_price < lo and pct_away <= proximity_pct:
                # Price is below zone and approaching from below → BOUNCE_SHORT
                setups.append({
                    "type": "BOUNCE_SHORT",
                    "zone_price": zp,
                    "zone_score": score,
                    "zone_levels": lvls,
                    "proximity_pct_away": round(pct_away * 100, 3),
                })

    return setups


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute 14-period Average True Range on daily bars."""
    hi = df["High"]
    lo = df["Low"]
    cl = df["Close"].shift(1)
    tr = pd.concat([hi - lo, (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


# ---------------------------------------------------------------------------
# Per-ticker scan
# ---------------------------------------------------------------------------

def scan_ticker(
    ticker: str,
    confluence_pct: float = DEFAULT_CONFLUENCE_PCT,
) -> dict[str, Any]:
    """Run the full mean-levels scan for a single ticker.

    Returns a result dict suitable for JSON serialisation.
    """
    result: dict[str, Any] = {
        "ticker": ticker,
        "scan_time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "error": None,
        "price": None,
        "levels": {},
        "atr": None,
        "zones": [],
        "setups": [],
    }

    try:
        df = fetch_ticker_data(ticker, period="3mo")
        levels = compute_mean_levels(df)
        current_price = float(df["Close"].iloc[-1])
        prev_price = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price
        atr = compute_atr(df)

        zones = find_confluence_zones(levels, confluence_pct)
        setups = detect_setups(current_price, prev_price, zones)

        result["price"] = round(current_price, 4)
        result["atr"] = round(atr, 4)
        result["levels"] = {k: (round(v, 4) if v is not None else None)
                             for k, v in levels.items()}
        result["zones"] = zones
        result["setups"] = setups
        result["top_setup"] = setups[0] if setups else None

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Batch scanner
# ---------------------------------------------------------------------------

def run_scanner(
    tickers: list[str],
    confluence_pct: float = DEFAULT_CONFLUENCE_PCT,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Scan a list of tickers and return all results.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to scan.
    confluence_pct : float
        Confluence band width (default 0.5 %).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    List of per-ticker result dicts sorted by number of setups (descending).
    """
    results: list[dict[str, Any]] = []

    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"[{i:>3}/{len(tickers)}] Scanning {ticker} ...", flush=True)
        result = scan_ticker(ticker, confluence_pct)
        results.append(result)

        if verbose and result["error"] is None:
            setups = result["setups"]
            zone_count = len(result["zones"])
            setup_str = ", ".join(s["type"] for s in setups) if setups else "—"
            print(f"         price={result['price']:.2f}  "
                  f"zones={zone_count}  setups=[{setup_str}]")
        elif verbose:
            print(f"         ERROR: {result['error']}")

    # Sort: tickers with active setups first, then by zone count
    results.sort(
        key=lambda r: (len(r["setups"]), len(r["zones"])),
        reverse=True,
    )
    return results


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a human-readable scan summary table."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("  MEAN LEVELS SCANNER – RESULTS SUMMARY")
    print(sep)
    print(f"  {'TICKER':<8} {'PRICE':>8}  {'CDM':>8}  {'PDM':>8}  "
          f"{'CMM':>8}  {'PMM':>8}  {'ZONES':>5}  SETUPS")
    print("-" * 80)

    for r in results:
        if r["error"]:
            print(f"  {r['ticker']:<8}  ERROR: {r['error']}")
            continue
        lvl = r["levels"]

        def fmt(v: float | None) -> str:
            return f"{v:>8.2f}" if v is not None else "     N/A"

        setup_str = " | ".join(
            f"{s['type']}@{s['zone_price']:.2f}(s={s['zone_score']})"
            for s in r["setups"]
        ) or "—"

        print(
            f"  {r['ticker']:<8} {r['price']:>8.2f} "
            f" {fmt(lvl.get('CDM'))} {fmt(lvl.get('PDM'))} "
            f" {fmt(lvl.get('CMM'))} {fmt(lvl.get('PMM'))} "
            f" {len(r['zones']):>5}  {setup_str}"
        )
    print(sep)

    active = [r for r in results if r["setups"]]
    print(f"\n  Tickers with active setups: {len(active)} / {len(results)}")
    if active:
        print("\n  TOP SETUPS:")
        for r in active[:10]:
            for s in r["setups"]:
                print(f"    {r['ticker']:<6} {s['type']:<14} zone={s['zone_price']:.2f} "
                      f"score={s['zone_score']} "
                      f"levels={'+'.join(s['zone_levels'])} "
                      f"({s['proximity_pct_away']:.2f}% away)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mean Levels S/R Scanner – compute CDM/PDM/CMM/PMM for equities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Ticker symbols to scan (default: 30 pre-defined names)",
    )
    parser.add_argument(
        "--confluence-pct", type=float, default=DEFAULT_CONFLUENCE_PCT,
        metavar="PCT",
        help=f"Confluence band as fraction of price (default {DEFAULT_CONFLUENCE_PCT})",
    )
    parser.add_argument(
        "--output", default="mean_levels_results.json", metavar="FILE",
        help="Output JSON file path (default: mean_levels_results.json)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Suppress progress output; print compact JSON to stdout instead",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tickers = args.tickers or DEFAULT_TICKERS
    verbose = not args.json

    if verbose:
        print(f"Mean Levels Scanner  |  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"Tickers : {len(tickers)}  |  Confluence band: {args.confluence_pct*100:.2f}%\n")

    results = run_scanner(tickers, confluence_pct=args.confluence_pct, verbose=verbose)

    output_payload = {
        "scan_time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "confluence_pct": args.confluence_pct,
        "ticker_count": len(results),
        "results": results,
    }

    if args.json:
        print(json.dumps(output_payload, separators=(",", ":")))
    else:
        print_summary(results)

    with open(args.output, "w") as fh:
        json.dump(output_payload, fh, indent=2)

    if verbose:
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
