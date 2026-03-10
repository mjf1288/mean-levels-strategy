#!/usr/bin/env python3
"""
Mean Levels S/R Scanner & Trader
=================================
Scans stocks for setups at daily/monthly mean levels and optionally executes
trades via Public.com.

Strategy: The 4 Mean Levels (from MeanTF Lua indicator — Lua-faithful engine)
  CDM — Current Day Mean     = running cumulative close average for today
  PDM — Previous Day Mean    = final cumulative close average of yesterday
  CMM — Current Month Mean   = running cumulative close average this calendar month
  PMM — Previous Month Mean  = final cumulative close average of last calendar month

These levels act as dynamic support/resistance. Confluence (multiple levels
stacking near each other) raises the significance of a zone.

Directional confirmation (hard gate filters on 8H bars = daily proxy):
  DSS Bressert  — Double Smoothed Stochastic (momentum oscillator)
  Lyapunov HP   — Hodrick-Prescott + Lyapunov divergence (trend regime)
  Both must agree with setup direction or the setup is killed.

Usage:
  python3 mean_levels_scanner.py                        # Scan default watchlist
  python3 mean_levels_scanner.py --tickers AAPL MSFT    # Scan specific tickers
  python3 mean_levels_scanner.py --execute              # Scan AND execute qualifying trades
  python3 mean_levels_scanner.py --equity 50000         # Override account equity
  python3 mean_levels_scanner.py --json                 # Machine-readable JSON output
  python3 mean_levels_scanner.py --dry-run              # Explicit dry-run (default behaviour)
"""

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------------

def _ensure(package: str, import_name: str = None) -> None:
    """Install *package* if it is not already importable."""
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

import pandas as pd          # noqa: E402 — imported after bootstrap
import yfinance as yf         # noqa: E402

# ---------------------------------------------------------------------------
# Directional indicators (DSS Bressert + Lyapunov HP)
# ---------------------------------------------------------------------------

import os
import importlib.util as _ilu

def _import_from_file(name: str, path: str):
    """Import a module from an absolute file path."""
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_WORKSPACE = os.path.dirname(os.path.abspath(__file__))
_dss_mod = _import_from_file("dss_bressert", os.path.join(_WORKSPACE, "dss_bressert.py"))
_lyap_mod = _import_from_file("lyapunov_hp", os.path.join(_WORKSPACE, "lyapunov_hp.py"))
DSSBressert = _dss_mod.DSSBressert
LyapunovHP = _lyap_mod.LyapunovHP

# ---------------------------------------------------------------------------
# Public.com SDK (optional — only needed for --execute)
# ---------------------------------------------------------------------------

_PUBLIC_SDK_AVAILABLE = False
try:
    from public_api_sdk import (          # type: ignore
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
        EquityMarketSession,
    )
    from public_api_sdk.auth_config import ApiKeyAuthConfig  # type: ignore
    _PUBLIC_SDK_AVAILABLE = True
except ImportError:
    pass   # SDK will be installed on demand inside execute_trades()

# Add the existing scripts directory to sys.path so we can import config.py
_SCRIPTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "skills", "publicdotcom-agent-skill",
    "publicdotcom-agent-skill", "scripts",
)
if os.path.isdir(_SCRIPTS_DIR) and _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def green(s: str)  -> str: return f"{_GREEN}{s}{_RESET}"
def red(s: str)    -> str: return f"{_RED}{s}{_RESET}"
def yellow(s: str) -> str: return f"{_YELLOW}{s}{_RESET}"
def cyan(s: str)   -> str: return f"{_CYAN}{s}{_RESET}"
def bold(s: str)   -> str: return f"{_BOLD}{s}{_RESET}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TICKERS: List[str] = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "TSLA", "AMD", "NFLX", "CRM", "AVGO", "JPM", "V",
    "MA", "UNH", "LLY", "COST",
]

# Weight assigned to each level for confluence scoring
LEVEL_WEIGHTS: Dict[str, int] = {"PMM": 4, "CMM": 3, "PDM": 2, "CDM": 1}

RESULTS_PATH = "/home/user/workspace/mean_levels_results.json"

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, days: int = 800) -> Optional[pd.DataFrame]:
    """
    Download *days* calendar-days of daily OHLCV data for *ticker* via yfinance.

    Default is 800 days (~3 years) to satisfy Lyapunov HP warmup (L_Period=525).

    Returns a DataFrame indexed by date (timezone-naive, ET), or None if the
    download fails or produces an empty result.
    """
    try:
        raw = yf.download(
            ticker,
            period=f"{days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            # suppress yfinance's own deprecation / info messages
        )
        if raw is None or raw.empty:
            return None

        # yfinance may return a MultiIndex if only one ticker — flatten
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Keep only standard OHLCV columns
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Drop rows where OHLC are all NaN (sometimes the last partial row)
        raw.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        # Ensure the index is timezone-naive (market data arrives in ET/UTC)
        if hasattr(raw.index, "tz") and raw.index.tz is not None:
            raw.index = raw.index.tz_convert("America/New_York").tz_localize(None)

        # Sort chronologically
        raw.sort_index(inplace=True)

        if len(raw) < 5:
            return None

        return raw

    except Exception as exc:
        print(yellow(f"  [warn] {ticker}: data fetch failed — {exc}"))
        return None


# ---------------------------------------------------------------------------
# Mean level computation  (Lua-faithful: running cumulative close average)
# ---------------------------------------------------------------------------

def _calc_mean_engine(closes: List[float], groups: List) -> Dict:
    """
    Lua-faithful mean engine.

    Iterates bar-by-bar computing a running cumulative average of *closes*
    within each *group* (trading date for daily, 'YYYY-MM' for monthly).
    When the group changes, the final running average of the prior group is
    captured as ``prev``.

    This exactly mirrors the MeanTF_dot.lua indicator logic:
      - buff          = cum / count  (running mean within the current TF candle)
      - prev_tf_buff  = buff at the last bar of the prior TF candle
      - direction     = 1 (UP) if buff rose vs the prior tick, -1 (DOWN) if it fell

    Parameters
    ----------
    closes : list of float — one close price per bar, chronological
    groups : list — group identifier per bar (same length as closes).
             For daily TF: the trading date (date object or string).
             For monthly TF: 'YYYY-MM' string.

    Returns
    -------
    dict with 'current' (float), 'prev' (float), 'dir' ('UP' or 'DOWN').
    """
    cum, count = 0.0, 0
    buff, prev_buff, prev_tf_buff = 0.0, 0.0, None
    direction = 0
    current_group = None

    for i in range(len(closes)):
        src = float(closes[i])
        grp = groups[i]

        if i == 0 or grp != current_group:
            # New TF candle — save the final mean of the prior candle
            if i > 0:
                prev_tf_buff = buff
            cum, count, current_group = src, 1, grp
        else:
            cum += src
            count += 1

        prev_buff = buff
        buff = cum / count

        if buff > prev_buff:
            direction = 1
        elif buff < prev_buff:
            direction = -1

    return {
        "current": round(buff, 6),
        "prev":    round(prev_tf_buff, 6) if prev_tf_buff is not None else round(buff, 6),
        "dir":     "UP" if direction == 1 else "DOWN",
    }


def compute_mean_levels(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the 4 mean levels from a daily OHLCV DataFrame using the
    Lua-faithful running cumulative close average engine.

    CDM (weight 1): running average of Close prices within the current trading day.
                    On daily bars this equals today's Close (one bar per day).
    PDM (weight 2): final running average of Close prices from the previous
                    trading day.  On daily bars this equals yesterday's Close.
    CMM (weight 3): running average of Close prices within the current calendar
                    month.  This is the cumulative mean of all daily closes so
                    far this month — it shifts as each new day is added.
    PMM (weight 4): final running average from the previous calendar month
                    (= simple average of all daily closes last month).

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLCV; index is DatetimeIndex (tz-naive), sorted chronologically.

    Returns
    -------
    dict with keys CDM, PDM, CMM, PMM (float values).
    """
    closes = df["Close"].tolist()

    # --- Daily groups: each trading date is its own group -----------------
    daily_groups = [
        (ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date())
        for ts in df.index
    ]
    daily = _calc_mean_engine(closes, daily_groups)
    cdm = daily["current"]    # running close avg of today (= today's close on daily bars)
    pdm = daily["prev"]       # final close avg of previous day (= yesterday's close)

    # --- Monthly groups: 'YYYY-MM' string per bar -------------------------
    monthly_groups = [
        (ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()).strftime("%Y-%m")
        for ts in df.index
    ]
    monthly = _calc_mean_engine(closes, monthly_groups)
    cmm = monthly["current"]  # running close avg so far this month
    pmm = monthly["prev"]     # final close avg of previous month

    return {"CDM": cdm, "PDM": pdm, "CMM": cmm, "PMM": pmm}


# ---------------------------------------------------------------------------
# Confluence zone computation
# ---------------------------------------------------------------------------

def compute_confluence_zones(
    levels: Dict[str, float],
    price: float,
    proximity_pct: float = 0.005,  # 0.5% grouping radius
) -> List[Dict]:
    """
    Group levels that are within *proximity_pct* of each other into zones
    and score them by their combined weights.

    Each zone dict contains:
      center        — price midpoint of the contributing levels
      score         — sum of LEVEL_WEIGHTS for contributing levels
      levels        — list of level names that make up the zone
      level_prices  — dict {name: price} for contributing levels

    Returns zones sorted by score (descending).
    """
    # Build a list of (price_value, name, weight) tuples
    items = [(v, k, LEVEL_WEIGHTS[k]) for k, v in levels.items()]

    assigned = [False] * len(items)
    zones: List[Dict] = []

    for i, (price_i, name_i, weight_i) in enumerate(items):
        if assigned[i]:
            continue

        group_prices  = [price_i]
        group_names   = [name_i]
        group_weights = [weight_i]
        group_map     = {name_i: price_i}
        assigned[i]   = True

        for j, (price_j, name_j, weight_j) in enumerate(items):
            if i == j or assigned[j]:
                continue
            # Two levels are "stacking" if they're within proximity_pct of each other
            if abs(price_i - price_j) / max(price_i, price_j) <= proximity_pct:
                group_prices.append(price_j)
                group_names.append(name_j)
                group_weights.append(weight_j)
                group_map[name_j] = price_j
                assigned[j] = True

        zone = {
            "center":       sum(group_prices) / len(group_prices),
            "score":        sum(group_weights),
            "levels":       group_names,
            "level_prices": group_map,
        }
        zones.append(zone)

    zones.sort(key=lambda z: z["score"], reverse=True)
    return zones


# ---------------------------------------------------------------------------
# Setup detection
# ---------------------------------------------------------------------------

def detect_setups(
    df: pd.DataFrame,
    levels: Dict[str, float],
    confluence_zones: List[Dict],
) -> List[Dict]:
    """
    Detect BOUNCE and BREAK setups against mean-level confluence zones.

    Setup types:
      BOUNCE_LONG  — price dropped to a support zone, showing reversal candle
      BOUNCE_SELL  — price rallied to a resistance zone, showing rejection candle
      BREAK_LONG   — price closed above a resistance zone after 2+ sessions below
      BREAK_SELL   — price closed below a support zone after 2+ sessions above

    Entry / Stop / Target logic
    ---------------------------
    BOUNCE_LONG:
      Entry  = current close (enter on next open, use close as proxy)
      Stop   = zone center * (1 - 0.005)  (0.5% below zone)
      T1     = entry + 1× risk
      T2     = entry + 2× risk
      T3     = entry + 3× risk

    BOUNCE_SELL (short / exit long):
      Entry  = current close
      Stop   = zone center * (1 + 0.005)
      T1–T3  mirror BOUNCE_LONG downward

    BREAK_LONG:
      Entry  = zone center  (buy the breakout retest at the level)
      Stop   = zone center * (1 - 0.007)
      T1–T3  same 1×/2×/3× risk multiples

    BREAK_SELL:
      Entry  = zone center
      Stop   = zone center * (1 + 0.007)
      T1–T3  mirror downward
    """
    if len(df) < 5:
        return []

    price        = float(df.iloc[-1]["Close"])
    recent_slice = df.iloc[-5:]
    recent_high  = float(recent_slice["High"].max())
    recent_low   = float(recent_slice["Low"].min())

    # How far has price stretched from its recent extremes?
    stretch_down = (recent_high - price) / recent_high  # positive when price < recent high
    stretch_up   = (price - recent_low)  / recent_low   # positive when price > recent low

    setups: List[Dict] = []

    for zone in confluence_zones:
        center    = zone["center"]
        score     = zone["score"]
        proximity = abs(price - center) / price  # fraction from current price to zone

        # ------------------------------------------------------------------
        # BOUNCE LONG — price near support, stretched down, green close above zone
        # ------------------------------------------------------------------
        if (
            proximity <= 0.003           # within 0.3% of zone
            and stretch_down >= 0.015    # dropped ≥1.5% from recent high
            and price >= center          # close is above (or at) the zone = reversal candle
            and score >= 3               # meaningful confluence
        ):
            stop   = center * 0.995
            risk   = abs(price - stop)
            setup  = {
                "type":     "BOUNCE_LONG",
                "zone":     zone,
                "score":    score,
                "entry":    round(price, 4),
                "stop":     round(stop, 4),
                "t1":       round(price + 1 * risk, 4),
                "t2":       round(price + 2 * risk, 4),
                "t3":       round(price + 3 * risk, 4),
                "risk_per_share": round(risk, 4),
            }
            setups.append(setup)

        # ------------------------------------------------------------------
        # BOUNCE SELL — price near resistance, stretched up, red close below zone
        # ------------------------------------------------------------------
        elif (
            proximity <= 0.003
            and stretch_up >= 0.015
            and price <= center          # close is below (or at) the zone = rejection
            and score >= 3
        ):
            stop   = center * 1.005
            risk   = abs(stop - price)
            setup  = {
                "type":     "BOUNCE_SELL",
                "zone":     zone,
                "score":    score,
                "entry":    round(price, 4),
                "stop":     round(stop, 4),
                "t1":       round(price - 1 * risk, 4),
                "t2":       round(price - 2 * risk, 4),
                "t3":       round(price - 3 * risk, 4),
                "risk_per_share": round(risk, 4),
            }
            setups.append(setup)

        # ------------------------------------------------------------------
        # BREAK LONG — price closed above zone after 2+ sessions below it
        # ------------------------------------------------------------------
        if len(df) >= 3 and price > center:
            prior_closes = [float(df.iloc[i]["Close"]) for i in [-2, -3]]
            sessions_below = sum(1 for c in prior_closes if c < center)
            if sessions_below >= 2 and score >= 3:
                stop   = center * 0.993
                risk   = abs(center - stop)
                setup  = {
                    "type":     "BREAK_LONG",
                    "zone":     zone,
                    "score":    score,
                    "entry":    round(center, 4),   # buy the retest of the broken level
                    "stop":     round(stop, 4),
                    "t1":       round(center + 1 * risk, 4),
                    "t2":       round(center + 2 * risk, 4),
                    "t3":       round(center + 3 * risk, 4),
                    "risk_per_share": round(risk, 4),
                }
                setups.append(setup)

        # ------------------------------------------------------------------
        # BREAK SELL — price closed below zone after 2+ sessions above it
        # ------------------------------------------------------------------
        if len(df) >= 3 and price < center:
            prior_closes = [float(df.iloc[i]["Close"]) for i in [-2, -3]]
            sessions_above = sum(1 for c in prior_closes if c > center)
            if sessions_above >= 2 and score >= 3:
                stop   = center * 1.007
                risk   = abs(stop - center)
                setup  = {
                    "type":     "BREAK_SELL",
                    "zone":     zone,
                    "score":    score,
                    "entry":    round(center, 4),
                    "stop":     round(stop, 4),
                    "t1":       round(center - 1 * risk, 4),
                    "t2":       round(center - 2 * risk, 4),
                    "t3":       round(center - 3 * risk, 4),
                    "risk_per_share": round(risk, 4),
                }
                setups.append(setup)

    # De-duplicate: if the same zone produced both a BOUNCE and BREAK setup, keep highest score
    seen_zone_types: set = set()
    unique_setups: List[Dict] = []
    for s in sorted(setups, key=lambda x: x["score"], reverse=True):
        key = (id(s["zone"]), s["type"])
        if key not in seen_zone_types:
            seen_zone_types.add(key)
            unique_setups.append(s)

    return unique_setups


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def calculate_position(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.01,
    score: int = 3,
) -> int:
    """
    Calculate share count using fixed fractional risk.

    Parameters
    ----------
    equity      : total account equity in dollars
    entry_price : intended entry price
    stop_price  : stop-loss price
    risk_pct    : fraction of equity to risk (default 1%)
    score       : confluence score — lower scores get half sizing

    Returns
    -------
    Number of shares (minimum 1).
    """
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share == 0:
        return 1

    risk_dollars = equity * risk_pct
    if score <= 3:
        risk_dollars *= 0.5   # half size for marginal confluence

    shares = int(risk_dollars / risk_per_share)
    return max(shares, 1)


# ---------------------------------------------------------------------------
# Per-ticker scan
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Directional indicator singletons (reused across all tickers)
# ---------------------------------------------------------------------------

_DSS_INDICATOR = DSSBressert()   # defaults: stoch=13, smooth=8, signal=8
_LYAP_INDICATOR = LyapunovHP()   # defaults: filter=7, L_Period=525


def compute_directional_signals(df: pd.DataFrame) -> Dict:
    """
    Compute DSS Bressert and Lyapunov HP directional signals.

    Returns dict with:
      dss_score, dss_direction, dss_value, dss_signal,
      lyap_score, lyap_direction, lyap_value,
      net_direction ('BULLISH', 'BEARISH', 'NEUTRAL'),
      gate_pass_long (bool), gate_pass_sell (bool)
    """
    # DSS Bressert
    dss_result = _DSS_INDICATOR.calculate(df)
    dss_score = _DSS_INDICATOR.get_signal_score(df)

    # Lyapunov HP
    lyap_result = _LYAP_INDICATOR.calculate_from_df(df)
    lyap_score = _LYAP_INDICATOR.get_signal_score(df)

    dss_dir = dss_result["direction"] if dss_result else "NEUTRAL"
    lyap_dir = lyap_result["direction"] if lyap_result else "NEUTRAL"

    # Net direction: both must agree for a strong signal
    if dss_dir == "BULLISH" and lyap_dir == "BULLISH":
        net = "BULLISH"
    elif dss_dir == "BEARISH" and lyap_dir == "BEARISH":
        net = "BEARISH"
    elif dss_dir == "NEUTRAL" and lyap_dir != "NEUTRAL":
        net = lyap_dir   # Lyapunov breaks tie
    elif lyap_dir == "NEUTRAL" and dss_dir != "NEUTRAL":
        net = dss_dir    # DSS breaks tie
    else:
        net = "NEUTRAL"  # conflicting or both neutral

    # Gate filter logic:
    #   LONG setups require net != BEARISH  (bullish or neutral pass)
    #   SELL setups require net != BULLISH  (bearish or neutral pass)
    gate_long = net != "BEARISH"
    gate_sell = net != "BULLISH"

    return {
        "dss_score":      dss_score,
        "dss_direction":  dss_dir,
        "dss_value":      dss_result["dss"] if dss_result else None,
        "dss_signal":     dss_result["signal"] if dss_result else None,
        "lyap_score":     lyap_score,
        "lyap_direction": lyap_dir,
        "lyap_value":     lyap_result["lyapunov"] if lyap_result else None,
        "net_direction":  net,
        "gate_pass_long": gate_long,
        "gate_pass_sell": gate_sell,
    }


def scan_ticker(ticker: str, equity: float) -> Optional[Dict]:
    """
    Run the full Mean Levels scan for a single *ticker*.

    Returns a dict with keys:
      ticker, price, levels, confluence_zones, setups, directional
    or None if data could not be fetched.
    """
    df = fetch_ohlcv(ticker, days=800)
    if df is None:
        return None

    levels  = compute_mean_levels(df)
    price   = float(df.iloc[-1]["Close"])
    zones   = compute_confluence_zones(levels, price)

    # Compute directional indicators (DSS + Lyapunov)
    directional = compute_directional_signals(df)

    # Detect raw setups
    raw_setups = detect_setups(df, levels, zones)

    # Apply directional hard gate filter
    setups = []
    gated_out = 0
    for setup in raw_setups:
        stype = setup["type"]
        if "LONG" in stype and not directional["gate_pass_long"]:
            gated_out += 1
            setup["gated"] = True
            setup["gate_reason"] = f"Direction BEARISH (DSS={directional['dss_direction']}, Lyap={directional['lyap_direction']})"
        elif "SELL" in stype and not directional["gate_pass_sell"]:
            gated_out += 1
            setup["gated"] = True
            setup["gate_reason"] = f"Direction BULLISH (DSS={directional['dss_direction']}, Lyap={directional['lyap_direction']})"
        else:
            setup["gated"] = False
            setup["gate_reason"] = None
            setups.append(setup)

    # Attach position sizing to passing setups
    for setup in setups:
        setup["shares"] = calculate_position(
            equity=equity,
            entry_price=setup["entry"],
            stop_price=setup["stop"],
            score=setup["score"],
        )
        setup["position_value"] = round(setup["shares"] * setup["entry"], 2)

    return {
        "ticker":            ticker,
        "price":             round(price, 4),
        "levels":            {k: round(v, 4) for k, v in levels.items()},
        "confluence_zones":  zones,
        "setups":            setups,
        "directional":       directional,
        "gated_count":       gated_out,
        "scan_time":         datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _setup_colour(setup_type: str) -> str:
    return green if "LONG" in setup_type else red  # type: ignore


def print_scan_results(results: List[Dict]) -> None:
    """Print a formatted table of all tickers and their setups."""
    print()
    print(bold("=" * 130))
    print(bold("  MEAN LEVELS S/R SCANNER + DIRECTIONAL FILTERS"))
    print(bold("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")))
    print(bold("=" * 130))

    # Summary of levels + directional signals for every ticker
    print()
    print(bold(
        f"{'TICKER':<8}  {'PRICE':>9}  {'CDM':>9}  {'PDM':>9}  "
        f"{'CMM':>9}  {'PMM':>9}  "
        f"{'DSS':>7} {'D-DIR':<8} {'LYAP':>8} {'L-DIR':<8} {'NET':<8}  {'SETUPS'}"
    ))
    print("-" * 130)

    all_setups: List[Tuple[str, Dict]] = []
    total_gated = 0

    for r in results:
        if r is None:
            continue
        lvl     = r["levels"]
        d       = r.get("directional", {})
        n_setup = len(r["setups"])
        gated   = r.get("gated_count", 0)
        total_gated += gated

        flag = ""
        if n_setup > 0:
            flag = green(f"{n_setup} found")
        if gated > 0:
            flag += f" {yellow(f'({gated} gated)')}"

        dss_val = f"{d.get('dss_value', 0) or 0:.1f}"
        dss_dir = d.get('dss_direction', 'N/A')[:4]
        lyap_val = f"{d.get('lyap_value', 0) or 0:.0f}"
        lyap_dir = d.get('lyap_direction', 'N/A')[:4]
        net_dir = d.get('net_direction', 'N/A')

        # Colour the net direction
        if net_dir == "BULLISH":
            net_str = green(f"{net_dir:<8}")
        elif net_dir == "BEARISH":
            net_str = red(f"{net_dir:<8}")
        else:
            net_str = yellow(f"{net_dir:<8}")

        print(
            f"{r['ticker']:<8}  "
            f"{r['price']:>9.2f}  "
            f"{lvl['CDM']:>9.2f}  "
            f"{lvl['PDM']:>9.2f}  "
            f"{lvl['CMM']:>9.2f}  "
            f"{lvl['PMM']:>9.2f}  "
            f"{dss_val:>7} {dss_dir:<8} {lyap_val:>8} {lyap_dir:<8} {net_str}  "
            f"{flag}"
        )
        for setup in r["setups"]:
            all_setups.append((r["ticker"], setup))

    print()
    if total_gated > 0:
        print(yellow(f"  {total_gated} setup(s) gated out by directional filters (DSS + Lyapunov)"))
        print()

    if not all_setups:
        print(yellow("  No setups passed all filters."))
        print()
        return

    # Detailed setup table
    print(bold("=" * 130))
    print(bold("  TRADE SETUPS (passed directional gate)"))
    print(bold("=" * 130))
    print()
    print(bold(
        f"{'TICKER':<7}  {'TYPE':<13}  {'SCORE':>5}  {'LEVELS':<16}  "
        f"{'ENTRY':>8}  {'STOP':>8}  {'T1':>8}  {'T2':>8}  {'T3':>8}  "
        f"{'SHARES':>6}  {'VALUE':>10}"
    ))
    print("-" * 120)

    for ticker, setup in sorted(all_setups, key=lambda x: x[1]["score"], reverse=True):
        colour   = _setup_colour(setup["type"])
        lvl_str  = "+".join(setup["zone"]["levels"])
        line     = (
            f"{ticker:<7}  "
            f"{setup['type']:<13}  "
            f"{setup['score']:>5}  "
            f"{lvl_str:<16}  "
            f"{setup['entry']:>8.2f}  "
            f"{setup['stop']:>8.2f}  "
            f"{setup['t1']:>8.2f}  "
            f"{setup['t2']:>8.2f}  "
            f"{setup['t3']:>8.2f}  "
            f"{setup['shares']:>6}  "
            f"${setup['position_value']:>9,.2f}"
        )
        print(colour(line))

    print()


# ---------------------------------------------------------------------------
# Trade execution via Public.com
# ---------------------------------------------------------------------------

def _ensure_sdk() -> None:
    """Install publicdotcom-py if not already available."""
    global _PUBLIC_SDK_AVAILABLE
    if _PUBLIC_SDK_AVAILABLE:
        return
    print("[bootstrap] Installing publicdotcom-py==0.1.8 …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "publicdotcom-py==0.1.8", "--quiet"],
        stdout=subprocess.DEVNULL,
    )
    # Re-import after installation
    from public_api_sdk import (          # type: ignore  # noqa: F401
        PublicApiClient, PublicApiClientConfiguration,
        OrderRequest, PreflightRequest, OrderInstrument,
        InstrumentType, OrderSide, OrderType,
        OrderExpirationRequest, TimeInForce, EquityMarketSession,
    )
    from public_api_sdk.auth_config import ApiKeyAuthConfig  # type: ignore  # noqa: F401
    _PUBLIC_SDK_AVAILABLE = True


def execute_trades(results: List[Dict]) -> None:
    """
    For each BOUNCE_LONG or BREAK_LONG setup, place a LIMIT BUY.
    For BOUNCE_SELL or BREAK_SELL, place a LIMIT SELL (short / exit).

    Runs preflight first and prints estimated cost before placing.
    """
    _ensure_sdk()

    from public_api_sdk import (          # type: ignore
        PublicApiClient, PublicApiClientConfiguration,
        OrderRequest, PreflightRequest, OrderInstrument,
        InstrumentType, OrderSide, OrderType,
        OrderExpirationRequest, TimeInForce,
    )
    from public_api_sdk.auth_config import ApiKeyAuthConfig  # type: ignore

    try:
        from config import get_api_secret, get_account_id  # type: ignore
    except ImportError:
        print(red("ERROR: Cannot import config.py. Check that SCRIPTS_DIR is correct."))
        return

    secret     = get_api_secret()
    account_id = get_account_id()

    if not secret:
        print(red("ERROR: PUBLIC_COM_SECRET environment variable is not set."))
        return
    if not account_id:
        print(red("ERROR: PUBLIC_COM_ACCOUNT_ID environment variable is not set."))
        return

    client = PublicApiClient(
        ApiKeyAuthConfig(api_secret_key=secret),
        config=PublicApiClientConfiguration(default_account_number=account_id),
    )

    executed = 0

    for result in results:
        if result is None:
            continue
        ticker = result["ticker"]

        for setup in result["setups"]:
            setup_type   = setup["type"]
            entry        = setup["entry"]
            shares       = setup["shares"]

            if setup_type in ("BOUNCE_LONG", "BREAK_LONG"):
                side = OrderSide.BUY
            elif setup_type in ("BOUNCE_SELL", "BREAK_SELL"):
                side = OrderSide.SELL
            else:
                continue

            side_str = "BUY" if side == OrderSide.BUY else "SELL"
            print()
            print(bold(f"--- Executing: {side_str} {shares} × {ticker} @ ${entry:.2f} [{setup_type}] ---"))

            # ------------------------------------------------------------------
            # Step 1: Preflight
            # ------------------------------------------------------------------
            try:
                pf_request = PreflightRequest(
                    instrument=OrderInstrument(
                        symbol=ticker,
                        type=InstrumentType.EQUITY,
                    ),
                    order_side=side,
                    order_type=OrderType.LIMIT,
                    quantity=Decimal(str(shares)),
                    limit_price=Decimal(str(entry)),
                    expiration=OrderExpirationRequest(time_in_force=TimeInForce.DAY),
                )
                pf_response = client.perform_preflight_calculation(pf_request)
                print(f"  Estimated cost : ${getattr(pf_response, 'estimated_total_cost', 'N/A')}")
                print(f"  Buying power Δ : ${getattr(pf_response, 'buying_power_impact', 'N/A')}")
            except Exception as exc:
                print(yellow(f"  [warn] Preflight failed: {exc} — continuing to place order anyway"))

            # ------------------------------------------------------------------
            # Step 2: Place order
            # ------------------------------------------------------------------
            try:
                order_id = str(uuid.uuid4())
                order_request = OrderRequest(
                    order_id=order_id,
                    instrument=OrderInstrument(
                        symbol=ticker,
                        type=InstrumentType.EQUITY,
                    ),
                    order_side=side,
                    order_type=OrderType.LIMIT,
                    quantity=shares,
                    limit_price=Decimal(str(entry)),
                    expiration=OrderExpirationRequest(time_in_force=TimeInForce.DAY),
                )
                response = client.place_order(order_request)
                print(green(f"  Order placed — ID: {response.order_id}"))
                setup["order_id"]     = response.order_id
                setup["order_status"] = "PLACED"
                executed += 1
            except Exception as exc:
                print(red(f"  ERROR placing order: {exc}"))
                setup["order_status"] = f"FAILED: {exc}"

    client.close()

    print()
    print(bold(f"Execution complete. {executed} order(s) placed."))


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(results: List[Dict], path: str = RESULTS_PATH, quiet: bool = False) -> None:
    """Serialise scan results to JSON, stripping non-serialisable objects."""

    def _clean(obj):
        """Recursively make obj JSON-serialisable."""
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        # Pandas Timestamp, numpy types, etc.
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)

    payload = {
        "scan_time": datetime.now().isoformat(),
        "results":   _clean(results),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)

    if not quiet:
        print(f"\nResults saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Tickers to scan (default: built-in watchlist of 20 names)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Place live orders for qualifying setups via Public.com",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Scan only, do not execute (default behaviour; alias for not using --execute)",
    )
    parser.add_argument(
        "--equity", type=float, default=None,
        help="Override account equity for position sizing (default: fetched from portfolio or 100,000)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print machine-readable JSON output instead of the formatted table",
    )
    parser.add_argument(
        "--risk-pct", type=float, default=1.0,
        help="Percentage of equity to risk per trade (default: 1.0)",
    )
    return parser


def resolve_equity(override: Optional[float], quiet: bool = False) -> float:
    """
    Return account equity to use for position sizing.
    1. Use --equity override if provided.
    2. Try to fetch from Public.com portfolio.
    3. Fall back to $100,000.

    Parameters
    ----------
    override : explicit override from --equity flag
    quiet    : suppress info messages (used in --json mode)
    """
    if override is not None:
        return override

    # Try to pull live equity from Public.com
    try:
        _ensure_sdk()
        from public_api_sdk import PublicApiClient, PublicApiClientConfiguration  # type: ignore
        from public_api_sdk.auth_config import ApiKeyAuthConfig                  # type: ignore
        from config import get_api_secret, get_account_id                        # type: ignore

        secret     = get_api_secret()
        account_id = get_account_id()
        if secret and account_id:
            client    = PublicApiClient(
                ApiKeyAuthConfig(api_secret_key=secret),
                config=PublicApiClientConfiguration(default_account_number=account_id),
            )
            portfolio = client.get_portfolio()
            total_equity = sum(e.value for e in portfolio.equity)
            client.close()
            if not quiet:
                print(f"[info] Account equity fetched from Public.com: ${total_equity:,.2f}")
            return float(total_equity)
    except Exception:
        pass   # silently fall through

    default = 100_000.0
    if not quiet:
        print(f"[info] Using default equity: ${default:,.2f} (pass --equity to override)")
    return default


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    tickers = args.tickers if args.tickers else DEFAULT_TICKERS
    equity  = resolve_equity(args.equity, quiet=args.json)

    if not args.json:
        print(cyan(f"\nScanning {len(tickers)} ticker(s)…"))

    results: List[Optional[Dict]] = []

    for i, ticker in enumerate(tickers, 1):
        ticker = ticker.upper()
        if not args.json:
            print(f"  [{i:>2}/{len(tickers)}] {ticker:<8}", end=" ", flush=True)
        result = scan_ticker(ticker, equity)
        if result is not None:
            n = len(result["setups"])
            if not args.json:
                label = green(f"{n} setup(s)") if n else "no setups"
                print(label)
        else:
            if not args.json:
                print(yellow("skipped (no data)"))
        results.append(result)

    valid_results = [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if args.json:
        # Machine-readable output
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            try:
                return float(obj)
            except (TypeError, ValueError):
                return str(obj)

        print(json.dumps({"scan_time": datetime.now().isoformat(), "results": _clean(valid_results)}, indent=2))
    else:
        print_scan_results(valid_results)

    # Always persist results to disk
    save_results(valid_results, quiet=args.json)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    if args.execute and not args.dry_run:
        print(bold("\n>>> EXECUTE MODE — placing orders <<<"))
        execute_trades(valid_results)
    elif args.execute and args.dry_run:
        print(yellow("\n[warn] --execute and --dry-run are mutually exclusive; skipping execution."))


if __name__ == "__main__":
    main()
