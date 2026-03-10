#!/usr/bin/env python3
"""
DSS Bressert — Double Smoothed Stochastic
==========================================
Faithful Python port of DSS_Bressert.lua indicator.

Algorithm (direct from Update() in Lua):
  1. Raw stochastic over `Frame` bars:
       HIGH = max(source.high, period-Frame+1 .. period)
       LOW  = min(source.low,  period-Frame+1 .. period)
       MIT  = (close - LOW) / (HIGH - LOW) * 100

  2. EMA smooth with SmoothCoefficient = 2/(1+EMAFrame):
       Buffer[i] = coeff * (MIT - Buffer[i-1]) + Buffer[i-1]

  3. Second stochastic on Buffer over (Frame+2) bars:
       LOW2, HIGH2 = minmax(Buffer, period-Frame-1 .. period)
       MIT2 = (Buffer[i] - LOW2) / (HIGH2 - LOW2) * 100

  4. Second EMA → DSS (called OUT in Lua):
       OUT[i] = coeff * (MIT2 - OUT[i-1]) + OUT[i-1]

  5. Signal = EMA of OUT using SignalFrame:
       Signal[i] = sig_coeff * (OUT[i] - Signal[i-1]) + Signal[i-1]

Lua defaults:  Frame=13, EMAFrame=8, SignalFrame=8, OB=80, OS=20

On 8-hour bars for equities, each trading day ≈ one 8H bar (the 9 AM open).
We use daily OHLCV data as the input.
"""

import math
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class DSSBressert:
    def __init__(
        self,
        stoch_period: int = 13,
        smooth_period: int = 8,
        signal_period: int = 8,
        ob_level: float = 80.0,
        os_level: float = 20.0,
    ):
        self.stoch_period = stoch_period      # Frame
        self.smooth_period = smooth_period    # EMAFrame
        self.signal_period = signal_period    # SignalFrame
        self.ob_level = ob_level
        self.os_level = os_level
        self.smooth_coeff = 2.0 / (1.0 + smooth_period)   # Lua: SmoothCoefficient
        self.signal_coeff = 2.0 / (1.0 + signal_period)

    def calculate(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Compute DSS and Signal from DataFrame with High, Low, Close columns.
        Returns None if insufficient data.
        """
        highs = df["High"].values.astype(float)
        lows = df["Low"].values.astype(float)
        closes = df["Close"].values.astype(float)
        n = len(closes)

        frame = self.stoch_period
        coeff = self.smooth_coeff

        # Lua warmup thresholds (0-indexed equivalents):
        #   first       = Frame           (step 1 starts here)
        #   2*Frame                       (step 3 starts here)
        #   3*Frame + EMAFrame            (DSS output starts here)
        #   + signal warmup               (Signal output starts here)
        first = frame
        min_bars = 3 * frame + self.smooth_period + self.signal_period + 2
        if n < min_bars:
            return None

        # ------------------------------------------------------------------
        # Step 1 + 2: Raw stochastic → EMA smooth → Buffer
        # ------------------------------------------------------------------
        buffer = np.zeros(n, dtype=float)

        for i in range(first, n):
            ws = i - frame + 1
            high = np.max(highs[ws:i + 1])
            low = np.min(lows[ws:i + 1])
            spread = high - low
            if spread == 0:
                mit = 50.0
            else:
                mit = (closes[i] - low) / spread * 100.0
            buffer[i] = coeff * (mit - buffer[i - 1]) + buffer[i - 1]

        # ------------------------------------------------------------------
        # Step 3 + 4: Second stochastic on Buffer → EMA smooth → OUT (DSS)
        # ------------------------------------------------------------------
        out = np.zeros(n, dtype=float)
        dss_start = 2 * frame  # Lua: "if period < 2*Frame then return"

        for i in range(dss_start, n):
            # Lua: mathex.minmax(Buffer, period-Frame-1, period)
            # That's from index (i - frame - 1) to i, inclusive = frame+2 bars
            bs = max(0, i - frame - 1)
            buf_slice = buffer[bs:i + 1]
            buf_low = np.min(buf_slice)
            buf_high = np.max(buf_slice)
            spread = buf_high - buf_low
            if spread == 0:
                mit2 = 50.0
            else:
                mit2 = (buffer[i] - buf_low) / spread * 100.0
            out[i] = coeff * (mit2 - out[i - 1]) + out[i - 1]

        # ------------------------------------------------------------------
        # Step 5: Signal = EMA of OUT
        # ------------------------------------------------------------------
        dss_valid = 3 * frame + self.smooth_period
        signal = np.zeros(n, dtype=float)
        signal[dss_valid] = out[dss_valid]
        for i in range(dss_valid + 1, n):
            signal[i] = self.signal_coeff * (out[i] - signal[i - 1]) + signal[i - 1]

        signal_valid = dss_valid + self.signal_period

        if signal_valid >= n:
            return None

        # ------------------------------------------------------------------
        # Build output
        # ------------------------------------------------------------------
        dss_arr = np.full(n, np.nan)
        sig_arr = np.full(n, np.nan)
        dss_arr[dss_valid:] = out[dss_valid:]
        sig_arr[signal_valid:] = signal[signal_valid:]

        latest_dss = float(dss_arr[-1])
        latest_sig = float(sig_arr[-1])
        prev_dss = float(dss_arr[-2]) if n >= 2 and not np.isnan(dss_arr[-2]) else None
        prev_sig = float(sig_arr[-2]) if n >= 2 and not np.isnan(sig_arr[-2]) else None

        direction = self._assess_direction(latest_dss, latest_sig, prev_dss, prev_sig)

        return {
            "dss": round(latest_dss, 4),
            "signal": round(latest_sig, 4),
            "prev_dss": round(prev_dss, 4) if prev_dss is not None else None,
            "prev_signal": round(prev_sig, 4) if prev_sig is not None else None,
            "direction": direction,
            "ob_level": self.ob_level,
            "os_level": self.os_level,
        }

    # ------------------------------------------------------------------
    # Direction assessment
    # ------------------------------------------------------------------

    def _assess_direction(self, dss, signal, prev_dss, prev_signal) -> str:
        if np.isnan(dss) or np.isnan(signal):
            return "NEUTRAL"
        if prev_dss is None or prev_signal is None:
            return "BULLISH" if dss > signal else "BEARISH" if dss < signal else "NEUTRAL"

        cross_up = prev_dss <= prev_signal and dss > signal
        cross_down = prev_dss >= prev_signal and dss < signal

        if cross_up:
            return "BULLISH"
        if cross_down:
            return "BEARISH"
        if dss > signal:
            return "BULLISH"
        if dss < signal:
            return "BEARISH"
        return "NEUTRAL"

    # ------------------------------------------------------------------
    # Scored signal for gate filter (-10 to +10)
    # ------------------------------------------------------------------

    def get_signal_score(self, df: pd.DataFrame) -> int:
        """
        Directional score from -10 to +10.

        +10 / -10: Fresh crossover in OS/OB zone (strongest)
        +7  / -7 : Fresh crossover in neutral zone
        +5  / -5 : DSS > signal, both in OS zone / DSS < signal, both in OB
        +3  / -3 : Sustained DSS > signal / DSS < signal
        +1  / -1 : Turning from extreme
         0      : Neutral / insufficient data
        """
        result = self.calculate(df)
        if result is None:
            return 0

        dss = result["dss"]
        sig = result["signal"]
        pd_ = result["prev_dss"]
        ps = result["prev_signal"]

        if pd_ is None or ps is None:
            return 0

        cross_up = pd_ <= ps and dss > sig
        cross_down = pd_ >= ps and dss < sig
        in_os = dss < self.os_level and sig < self.os_level
        in_ob = dss > self.ob_level and sig > self.ob_level

        if cross_up and in_os:
            return 10
        if cross_down and in_ob:
            return -10
        if cross_up:
            return 7
        if cross_down:
            return -7
        if dss > sig and in_os:
            return 5
        if dss < sig and in_ob:
            return -5
        if dss > sig:
            return 3
        if dss < sig:
            return -3
        if dss < self.os_level and dss > pd_:
            return 1
        if dss > self.ob_level and dss < pd_:
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
    print(f"DSS Bressert on {ticker} (daily bars as 8H proxy)...")

    raw = yf.download(ticker, period="120d", interval="1d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    dss = DSSBressert()
    result = dss.calculate(raw)
    if result:
        print(f"  DSS:       {result['dss']}")
        print(f"  Signal:    {result['signal']}")
        print(f"  Direction: {result['direction']}")
        print(f"  Score:     {dss.get_signal_score(raw)}")
    else:
        print("  Not enough data")
