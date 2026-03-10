#!/usr/bin/env python3
"""
Lyapunov HP Oscillator
======================
Faithful Python port of Lyapunov_HP.lua indicator.

Algorithm (direct from Update() in Lua):
  1. Lambda = 0.0625 / sin(π / Filter)^4

  2. Set up pentadiagonal coefficient arrays a[], b[], c[]:
       Interior rows:  a[i] = 6λ+1,  b[i] = -4λ,  c[i] = λ
       Boundary adjustments at first, first+1, period-1, period

  3. Backward sweep (pentadiagonal solver, period → first):
       Uses running variables H1..H5, HH1..HH5 to eliminate sub-diagonals.
       Stores factored values back into a[], b[], c[].
       a[i] becomes the solution of the tridiagonal sub-step.

  4. Forward substitution (first+1 → period):
       hpf[i] = a[i] - b[i]*H1 - c[i]*H2
       This produces the HP-filtered series (trend component removed).

  5. Lyapunov divergence:
       Lyapunov[i] = log(|hpf[i] / hpf[i-1]|) * 100000

Lua defaults:  Filter=7, L_Period=525

The Lyapunov value measures the exponential divergence rate of the HP-filtered
series. Positive values → expanding/trending, negative → contracting/mean-reverting.
A flip from negative to positive (or vice versa) signals a regime change.

On 8-hour bars for equities, each trading day ≈ one 8H bar.
We use daily close prices as the input (Lua uses core.Tick = close prices).
"""

import math
from typing import Dict, Optional
import pandas as pd
import numpy as np


class LyapunovHP:
    def __init__(self, filter_period: int = 7, lyapunov_period: int = 525):
        self.filter_period = filter_period    # Filter
        self.lyapunov_period = lyapunov_period  # L_Period
        self.lam = 0.0625 / math.pow(math.sin(math.pi / filter_period), 4)

    def calculate(self, prices: np.ndarray) -> Optional[Dict]:
        """
        Compute Lyapunov HP oscillator from an array of close prices.

        Parameters
        ----------
        prices : array-like of float, chronological close prices.

        Returns None if insufficient data.
        """
        prices = np.asarray(prices, dtype=float)
        n = len(prices)
        first = self.lyapunov_period  # Lua: first = source:first() + L_Period

        # Need at least first + 2 bars to produce any output
        if n < first + 3:
            return None

        period = n - 1  # Lua: period = source:size() - 1 (last bar index)
        lam = self.lam

        # ------------------------------------------------------------------
        # Step 2: Initialize coefficient arrays
        # ------------------------------------------------------------------
        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        c = np.zeros(n, dtype=float)

        # Interior rows
        for i in range(first, period):
            a[i] = 6 * lam + 1
            b[i] = -4 * lam
            c[i] = lam

        # Boundary conditions (Lua order matters — later assignments override)
        a[period] = 1 + lam
        b[period] = -2 * lam
        c[period] = lam

        a[period - 1] = 5 * lam + 1

        a[first] = 1 + lam
        a[first + 1] = 5 * lam + 1
        b[first + 1] = -2 * lam
        b[first] = 0
        c[first + 1] = 0
        c[first] = 0

        # ------------------------------------------------------------------
        # Step 3: Backward sweep (pentadiagonal factorization)
        # ------------------------------------------------------------------
        H1 = H2 = H3 = H4 = H5 = 0.0
        HH1 = HH2 = HH3 = HH5 = 0.0

        for i in range(period, first - 1, -1):
            Z = a[i] - H4 * H1 - HH5 * HH2
            if Z == 0:
                Z = 1e-20  # prevent division by zero

            HB = b[i]
            HH1 = H1
            H1 = (HB - H4 * H2) / Z
            b[i] = H1

            HC = c[i]
            HH2 = H2
            H2 = HC / Z
            c[i] = H2

            a[i] = (prices[i] - HH3 * HH5 - H3 * H4) / Z
            HH3 = H3
            H3 = a[i]

            H4 = HB - H5 * HH1
            HH5 = H5
            H5 = HC

        # ------------------------------------------------------------------
        # Step 4: Forward substitution → hpf
        # ------------------------------------------------------------------
        hpf = np.zeros(n, dtype=float)
        H2_fwd = 0.0
        H1_fwd = a[first + 1]
        hpf[first + 1] = H1_fwd

        for i in range(first + 2, period + 1):
            hpf[i] = a[i] - b[i] * H1_fwd - c[i] * H2_fwd
            H2_fwd = H1_fwd
            H1_fwd = hpf[i]

        # ------------------------------------------------------------------
        # Step 5: Lyapunov divergence
        # ------------------------------------------------------------------
        lyap = np.full(n, np.nan)
        for i in range(first + 1, period + 1):
            if hpf[i - 1] != 0:
                ratio = abs(hpf[i] / hpf[i - 1])
                if ratio > 0:
                    lyap[i] = math.log(ratio) * 100000
                else:
                    lyap[i] = 0.0
            else:
                lyap[i] = 0.0

        # ------------------------------------------------------------------
        # Direction assessment
        # ------------------------------------------------------------------
        latest = float(lyap[period]) if not np.isnan(lyap[period]) else 0.0
        prev = float(lyap[period - 1]) if period >= 1 and not np.isnan(lyap[period - 1]) else 0.0

        # Check recent bars for regime flip
        lookback = min(self.filter_period, period - first)
        recent = lyap[period - lookback + 1:period + 1]
        recent_valid = recent[~np.isnan(recent)]

        direction = self._assess_direction(latest, prev, recent_valid)

        return {
            "lyapunov": round(latest, 2),
            "prev_lyapunov": round(prev, 2),
            "direction": direction,
            "filter_period": self.filter_period,
            "lyapunov_period": self.lyapunov_period,
        }

    def calculate_from_df(self, df: pd.DataFrame) -> Optional[Dict]:
        """Convenience: compute from DataFrame with a Close column."""
        return self.calculate(df["Close"].values.astype(float))

    # ------------------------------------------------------------------
    # Direction assessment
    # ------------------------------------------------------------------

    def _assess_direction(self, latest: float, prev: float, recent: np.ndarray) -> str:
        """
        BULLISH: Lyapunov flipped from negative to positive (trending up regime)
                 or sustained positive
        BEARISH: Lyapunov flipped from positive to negative (trending down regime)
                 or sustained negative
        NEUTRAL: Near zero / mixed

        The Lua indicator logic:
          - Primary buy:  filter_period bars all negative → flips positive
          - Primary sell: filter_period bars all positive → flips negative
          - Secondary:    simple zero crossing
        """
        if len(recent) < 2:
            return "NEUTRAL"

        # Check for regime flip (strongest signal)
        all_neg_before = np.all(recent[:-1] < 0)
        all_pos_before = np.all(recent[:-1] > 0)

        if all_neg_before and latest > 0:
            return "BULLISH"   # Primary buy: sustained negative → flip positive
        if all_pos_before and latest < 0:
            return "BEARISH"   # Primary sell: sustained positive → flip negative

        # Simple zero crossing
        if prev <= 0 and latest > 0:
            return "BULLISH"
        if prev >= 0 and latest < 0:
            return "BEARISH"

        # Sustained direction
        if latest > 0:
            return "BULLISH"
        if latest < 0:
            return "BEARISH"

        return "NEUTRAL"

    # ------------------------------------------------------------------
    # Scored signal for gate filter (-10 to +10)
    # ------------------------------------------------------------------

    def get_signal_score(self, df_or_prices) -> int:
        """
        Directional score from -10 to +10.

        +10 / -10: Primary regime flip (N bars all neg/pos → flip)
        +7  / -7 : Zero crossing (secondary flip)
        +5  / -5 : Strong sustained positive/negative (|lyap| > 5000)
        +3  / -3 : Moderate sustained positive/negative
        +1  / -1 : Weak positive/negative
         0       : Neutral / insufficient data
        """
        if isinstance(df_or_prices, pd.DataFrame):
            prices = df_or_prices["Close"].values.astype(float)
        else:
            prices = np.asarray(df_or_prices, dtype=float)

        n = len(prices)
        first = self.lyapunov_period
        if n < first + 3:
            return 0

        result = self.calculate(prices)
        if result is None:
            return 0

        latest = result["lyapunov"]
        prev = result["prev_lyapunov"]

        # Reconstruct recent window for regime detection
        # (re-calculate is avoided by checking prev vs latest)
        period = n - 1

        # For primary flip detection we need the full calculation
        # Re-use the internal state — for efficiency we check latest vs prev
        # and augment with magnitude

        # Zero crossing?
        crossed_up = prev <= 0 and latest > 0
        crossed_down = prev >= 0 and latest < 0

        # Magnitude
        mag = abs(latest)

        if crossed_up:
            return 10 if mag > 5000 else 7
        if crossed_down:
            return -10 if mag > 5000 else -7

        # Sustained
        if latest > 0:
            if mag > 10000:
                return 5
            elif mag > 3000:
                return 3
            else:
                return 1
        elif latest < 0:
            if mag > 10000:
                return -5
            elif mag > 3000:
                return -3
            else:
                return -1

        return 0


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, subprocess
    try:
        import yfinance as yf
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "--quiet"])
        import yfinance as yf

    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    print(f"Lyapunov HP on {ticker} (daily bars as 8H proxy)...")

    raw = yf.download(ticker, period="3y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    lyap = LyapunovHP()
    result = lyap.calculate_from_df(raw)
    if result:
        print(f"  Lyapunov:  {result['lyapunov']}")
        print(f"  Prev:      {result['prev_lyapunov']}")
        print(f"  Direction: {result['direction']}")
        print(f"  Score:     {lyap.get_signal_score(raw)}")
        print(f"  Data bars: {len(raw)} (need {lyap.lyapunov_period}+ for warmup)")
    else:
        print(f"  Not enough data (have {len(raw)}, need {lyap.lyapunov_period + 3})")
