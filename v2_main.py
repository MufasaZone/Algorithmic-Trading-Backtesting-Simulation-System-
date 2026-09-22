"""
main.py -- TrendStructure + PDZones filter + MQL5-exact Japanese pattern scanner + Trade Simulator
Perfect port of the MQL5 TrendStructure_PDZones indicator.
With SMA Trend Strength Oscillator filter and MaxFavR_b4SL metric.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import pandas as pd
import numpy as np

import v2_config as config


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def map_timeframe(ctf_str: str) -> str:
    mapping = {
        "1min": "5min", "5min": "10min", "15min": "30min", "30min": "1h",
        "1h": "4h", "2h": "4h", "3h": "4h", "4h": "1D", "1D": "1D",
    }
    return mapping.get(ctf_str, "1D")

def detect_csv_timeframe(df: pd.DataFrame) -> pd.Timedelta:
    if len(df) < 2:
        raise ValueError("Need at least 2 rows")
    deltas = df["time"].diff().dropna()
    native = deltas.mode().iloc[0]
    if native <= pd.Timedelta(0):
        raise ValueError(f"Non‑positive spacing {native}")
    return native

def format_timeframe(delta: pd.Timedelta) -> str:
    total = int(delta.total_seconds())
    if total <= 0:
        return str(delta)
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if s == 0 and m == 0 and h == 0 and d > 0:
        return f"{d}D"
    if s == 0 and m == 0 and d == 0:
        return f"{h}h"
    if s == 0 and d == 0 and h == 0:
        return f"{m}min"
    return str(delta)

def resample_to_tf(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    target = pd.Timedelta(target_tf)
    agg = df.set_index("time").resample(target, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return agg.dropna(subset=["open", "high", "low", "close"]).reset_index()


# ---------------------------------------------------------------------------
# Pre‑compute ATR
# ---------------------------------------------------------------------------

def precompute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    high = df["high"].values
    low  = df["low"].values
    close = df["close"].values
    n = len(df)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i]  - close[i-1]))
    # MQL5's PDZ_CalculateATR is only ever called when i >= InpATRPeriod
    # (see PDZ_RunCalculate: `if(InpUseATR && i >= InpATRPeriod)`), so the
    # first bar that uses an ATR-based buffer is index == period, not
    # period - 1. pandas' rolling(period).mean() naturally produces its
    # first valid value at period - 1 (a full window ending there), which
    # is one bar too early -- shift the valid region forward by one bar so
    # bar `period - 1` still falls back to the fixed buffer_points, exactly
    # like MQL5.
    atr = np.full(n, np.nan)
    if n > period:
        rolled = pd.Series(tr).rolling(period).mean().values
        atr[period:] = rolled[period:]
    return atr


# ---------------------------------------------------------------------------
# Trend engines – exact MQL5 port
# ---------------------------------------------------------------------------

def is_pivot_high(high, i, length, n):
    if i - length < 0 or i + length >= n: return False
    v = high[i]
    for k in range(1, length+1):
        if high[i-k] > v or high[i+k] > v: return False
    return True

def is_pivot_low(low, i, length, n):
    if i - length < 0 or i + length >= n: return False
    v = low[i]
    for k in range(1, length+1):
        if low[i-k] < v or low[i+k] < v: return False
    return True

def compute_trend_pivot_pairs(o, h, l, c, pivot_len, req_pairs, hold_last):
    n = len(c)
    trend = np.full(n, np.nan, dtype=float)
    if n < pivot_len*2+5:
        return trend

    hh = ll = lh = hl = 0
    last_ph = last_pl = 0.0
    have_ph = have_pl = False
    current_trend = 0
    pending_at, pending_val = [], []
    apply_cursor = 0
    max_eval = n - pivot_len - 2

    for i in range(max_eval+1):
        ph = is_pivot_high(h, i, pivot_len, n)
        pl = is_pivot_low(l, i, pivot_len, n)
        if ph:
            new_high = h[i]
            is_higher = not have_ph or new_high > last_ph
            is_lower  = have_ph and new_high < last_ph
            last_ph = new_high; have_ph = True
            if is_higher: hh = min(hh+1, req_pairs); lh = 0
            elif is_lower: lh = min(lh+1, req_pairs); hh = 0
        if pl:
            new_low = l[i]
            is_higher = not have_pl or new_low > last_pl
            is_lower  = have_pl and new_low < last_pl
            last_pl = new_low; have_pl = True
            if is_higher: hl = min(hl+1, req_pairs); ll = 0
            elif is_lower: ll = min(ll+1, req_pairs); hl = 0
        if ph or pl:
            bull = (hh>=req_pairs) and (hl>=req_pairs)
            bear = (lh>=req_pairs) and (ll>=req_pairs)
            nt = current_trend
            if bull and not bear: nt = 1
            elif bear and not bull: nt = -1
            elif not hold_last: nt = 0
            if nt != current_trend:
                pending_at.append(i + pivot_len)
                pending_val.append(nt)
            current_trend = nt

    applied = 0
    for idx in range(max_eval+1):
        while apply_cursor < len(pending_at) and pending_at[apply_cursor] == idx:
            applied = pending_val[apply_cursor]
            apply_cursor += 1
        trend[idx] = applied
    return trend

def compute_trend_bos_choch(h, l, c, pivot_len, min_pivots=1):
    """Structural confirmation: a pivot on the side that WOULD break the
    current trend (opposite the current trend's defining side) counts
    toward the next flip's requirement. `min_pivots` of them must confirm
    since the last flip before the next flip is allowed to fire.
    min_pivots=1 reproduces the original (unconstrained) behavior."""
    n = len(c)
    trend = np.full(n, np.nan, dtype=float)
    if n < pivot_len*2+5:
        return trend

    sh = sl = 0.0
    have_sh = have_sl = False
    cur = 0
    piv_at, piv_val, piv_is_high = [], [], []
    piv_cursor = 0
    max_eval = n - pivot_len - 2
    same_side_pivot_count = 0

    for i in range(max_eval+1):
        ph = is_pivot_high(h, i, pivot_len, n)
        pl = is_pivot_low(l, i, pivot_len, n)
        if ph:
            piv_at.append(i+pivot_len); piv_val.append(h[i]); piv_is_high.append(True)
        if pl:
            piv_at.append(i+pivot_len); piv_val.append(l[i]); piv_is_high.append(False)
        while piv_cursor < len(piv_at) and piv_at[piv_cursor] == i:
            this_is_high = piv_is_high[piv_cursor]
            if this_is_high: sh = piv_val[piv_cursor]; have_sh = True
            else: sl = piv_val[piv_cursor]; have_sl = True

            # cur <= 0 (flat/bear) -> waiting on a bull flip -> breaking side is highs
            # cur >= 0 (flat/bull) -> waiting on a bear flip -> breaking side is lows
            if cur <= 0 and this_is_high:
                same_side_pivot_count += 1
            if cur >= 0 and not this_is_high:
                same_side_pivot_count += 1

            piv_cursor += 1

        if cur <= 0 and have_sh and c[i] > sh and same_side_pivot_count >= min_pivots:
            cur = 1
            same_side_pivot_count = 0
        elif cur >= 0 and have_sl and c[i] < sl and same_side_pivot_count >= min_pivots:
            cur = -1
            same_side_pivot_count = 0
        trend[i] = cur

    return trend

def compute_trend_choch_then_bos(h, l, c, pivot_len, hold_last, min_pivots=1):
    """Structural confirmation: BOS may only confirm a pending CHoCH once
    at least `min_pivots` opposite-side pivots have confirmed since the
    CHoCH trigger (evidence the opposite leg/base is actually building).
    min_pivots=1 reproduces the original (unconstrained) behavior."""
    n = len(c)
    trend = np.full(n, np.nan, dtype=float)
    if n < pivot_len*2+5:
        return trend

    sh = sl = 0.0
    have_sh = have_sl = False
    confirmed = 0
    pending_dir = 0
    choch_ref = 0.0
    opposite_pivot_count = 0

    piv_at, piv_val, piv_is_high = [], [], []
    piv_cursor = 0
    max_eval = n - pivot_len - 2

    for i in range(max_eval+1):
        ph = is_pivot_high(h, i, pivot_len, n)
        pl = is_pivot_low(l, i, pivot_len, n)
        if ph:
            piv_at.append(i+pivot_len); piv_val.append(h[i]); piv_is_high.append(True)
        if pl:
            piv_at.append(i+pivot_len); piv_val.append(l[i]); piv_is_high.append(False)
        while piv_cursor < len(piv_at) and piv_at[piv_cursor] == i:
            this_is_high = piv_is_high[piv_cursor]
            if this_is_high: sh = piv_val[piv_cursor]; have_sh = True
            else: sl = piv_val[piv_cursor]; have_sl = True

            # pending_dir == 1 (bullish CHoCH pending) waits on the up-leg
            # to prove itself, so each new confirmed swing LOW forming
            # during that wait is one unit of "the base is building"
            # evidence. Symmetric for pending_dir == -1 / swing highs.
            if pending_dir == 1 and not this_is_high:
                opposite_pivot_count += 1
            elif pending_dir == -1 and this_is_high:
                opposite_pivot_count += 1

            piv_cursor += 1

        close_i = c[i]

        if pending_dir == 0:
            if confirmed <= 0 and have_sh and close_i > sh:
                pending_dir = 1
                choch_ref = sh
                opposite_pivot_count = 0
            elif confirmed >= 0 and have_sl and close_i < sl:
                pending_dir = -1
                choch_ref = sl
                opposite_pivot_count = 0
        elif pending_dir == 1:
            if have_sl and close_i < sl:
                pending_dir = 0
            elif have_sh and close_i > sh and sh != choch_ref and opposite_pivot_count >= min_pivots:
                confirmed = 1
                pending_dir = 0
        elif pending_dir == -1:
            if have_sh and close_i > sh:
                pending_dir = 0
            elif have_sl and close_i < sl and sl != choch_ref and opposite_pivot_count >= min_pivots:
                confirmed = -1
                pending_dir = 0

        if pending_dir != 0 and not hold_last:
            painted_trend = 0
        else:
            painted_trend = confirmed
        trend[i] = painted_trend

    return trend

def compute_trend(df, engine, pivot_len, req_pairs=2, hold_last=False, min_pivots=1):
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    if engine == "PIVOT_PAIRS":
        return compute_trend_pivot_pairs(o, h, l, c, pivot_len, req_pairs, hold_last)
    elif engine == "BOS_CHOCH":
        return compute_trend_bos_choch(h, l, c, pivot_len, min_pivots)
    elif engine == "CHOCH_THEN_BOS":
        return compute_trend_choch_then_bos(h, l, c, pivot_len, hold_last, min_pivots)
    else:
        raise ValueError(f"Unknown engine: {engine}")


# ---------------------------------------------------------------------------
# PD Zones – exact MQL5 port
# ---------------------------------------------------------------------------

def compute_pd_zones_vectorized(df, lookback, use_atr, atr_period, atr_mult,
                                buffer_points, use_manual,
                                disc_inner, disc_outer, prem_inner, prem_outer,
                                point_size, atr_array=None):
    h = df["high"].values; l = df["low"].values
    n = len(df)
    roll_max = pd.Series(h).rolling(lookback, min_periods=lookback).max().values
    roll_min = pd.Series(l).rolling(lookback, min_periods=lookback).min().values
    mid = (roll_max + roll_min) / 2.0

    if use_manual:
        dt = mid - disc_inner * point_size
        db = mid - disc_outer * point_size
        pb = mid + prem_inner * point_size
        pt = mid + prem_outer * point_size
    else:
        if use_atr and atr_array is not None:
            buf = atr_array * atr_mult
            buf = np.where(np.isnan(buf), buffer_points * point_size, buf)
        else:
            buf = np.full(n, buffer_points * point_size)
        dt = mid - buf
        db = mid - buf * 3
        pb = mid + buf
        pt = mid + buf * 3

    valid = ~np.isnan(roll_max)
    eq = np.where(valid, mid, np.nan)
    prem_top = np.where(valid, pt, np.nan)
    prem_bot = np.where(valid, pb, np.nan)
    disc_top = np.where(valid, dt, np.nan)
    disc_bot = np.where(valid, db, np.nan)

    # Inner buffer distances (not absolute price levels) -- these are what
    # PDZ_RunCalculate calls bufferSizeDiscount / bufferSizePremium and
    # compares confluence/reversal thresholds against. Manual mode uses the
    # inner offsets directly (matches disc_inner/prem_inner in points); ATR
    # / fixed-points mode uses the same buf array used for dt/pb above.
    if use_manual:
        buf_discount = np.full(n, disc_inner * point_size)
        buf_premium = np.full(n, prem_inner * point_size)
    else:
        buf_discount = buf
        buf_premium = buf

    return eq, prem_top, prem_bot, disc_top, disc_bot, valid, buf_discount, buf_premium


def _pdz_is_bullish_reversal(o, h, l, c, i):
    """Port of PDZ_IsBullishReversal: bullish candle with body >40% of range."""
    if c[i] <= o[i]:
        return False
    body = c[i] - o[i]
    rng = h[i] - l[i]
    if rng == 0:
        return False
    return (body / rng) > 0.4


def _pdz_is_bearish_reversal(o, h, l, c, i):
    """Port of PDZ_IsBearishReversal: bearish candle with body >40% of range."""
    if c[i] >= o[i]:
        return False
    body = o[i] - c[i]
    rng = h[i] - l[i]
    if rng == 0:
        return False
    return (body / rng) > 0.4


def compute_pd_zone_signals(df, eq, buf_discount, buf_premium, point_size,
                            confluence_lookback, min_impulse_points):
    """Exact port of the InpShowSignals block in PDZ_RunCalculate: confluence-
    gated Buy/Sell arrow signals at discount/premium zone extremes.

    Buy fires when price is in discount territory (close < mid - bufDiscount)
    AND either (a) the current low re-tests/undercuts the recent swing low by
    <= 1 point while closing back above it, or (b) a bearish impulse move
    within the lookback window reversed upward by >= min_impulse_points,
    AND the current candle is a bullish reversal candle (body > 40% of range).
    Sell is the exact mirror. Matches MQL5's `i < rates_total - 1` guard --
    the live/still-forming bar never gets a signal.

    Returns (buy_signal, sell_signal) boolean arrays, same length as df.
    """
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    n = len(df)

    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)

    for i in range(n - 1):  # MQL5: `if(!InpShowSignals || i >= rates_total - 1) continue;`
        if np.isnan(eq[i]):
            continue
        midpoint = eq[i]
        buf_disc = buf_discount[i]
        buf_prem = buf_premium[i]

        # --- Buy: price in discount territory ---
        if c[i] < midpoint - buf_disc:
            confluence = False

            recent_low = l[i]
            for j in range(1, confluence_lookback + 1):
                if i - j < 0:
                    break
                if l[i - j] < recent_low:
                    recent_low = l[i - j]

            if l[i] <= recent_low + point_size and c[i] > recent_low:
                confluence = True

            if not confluence:
                for j in range(2, confluence_lookback + 1):
                    if i - j < 0:
                        break
                    if c[i - j] >= o[i - j]:
                        continue
                    impulse = 0.0
                    for k in range(1, 4):
                        if i - j + k > i:
                            break
                        move = c[i - j + k] - c[i - j]
                        if move > impulse:
                            impulse = move
                    if impulse >= min_impulse_points * point_size:
                        confluence = True
                        break

            if confluence and _pdz_is_bullish_reversal(o, h, l, c, i):
                buy_signal[i] = True

        # --- Sell: price in premium territory ---
        if c[i] > midpoint + buf_prem:
            confluence = False

            recent_high = h[i]
            for j in range(1, confluence_lookback + 1):
                if i - j < 0:
                    break
                if h[i - j] > recent_high:
                    recent_high = h[i - j]

            if h[i] >= recent_high - point_size and c[i] < recent_high:
                confluence = True

            if not confluence:
                for j in range(2, confluence_lookback + 1):
                    if i - j < 0:
                        break
                    if c[i - j] <= o[i - j]:
                        continue
                    impulse = 0.0
                    for k in range(1, 4):
                        if i - j + k > i:
                            break
                        move = c[i - j] - c[i - j + k]
                        if move > impulse:
                            impulse = move
                    if impulse >= min_impulse_points * point_size:
                        confluence = True
                        break

            if confluence and _pdz_is_bearish_reversal(o, h, l, c, i):
                sell_signal[i] = True

    return buy_signal, sell_signal


def map_htf_zones_to_ctf(htf_df, ctf_df, htf_zone_arrays):
    eq, pt, pb, dt, db, valid = htf_zone_arrays
    htf_times = htf_df["time"].values
    ctf_times = ctf_df["time"].values

    if len(htf_df) > 0:
        htf_delta = pd.Timedelta(htf_df["time"].diff().mode().iloc[0])
        last_htf_start = htf_times[-1]
        if last_htf_start + htf_delta > ctf_times[-1]:
            if len(htf_df) > 1:
                htf_times = htf_times[:-1]
                eq = eq[:-1]
                pt = pt[:-1]
                pb = pb[:-1]
                dt = dt[:-1]
                db = db[:-1]
                valid = valid[:-1]
            else:
                return (np.full(len(ctf_df), np.nan),) * 5 + (np.full(len(ctf_df), False),)

    idx = np.searchsorted(htf_times, ctf_times, side='right') - 1
    idx = np.clip(idx, 0, len(eq)-1)

    mapped_eq = np.where(idx >= 0, eq[idx], np.nan)
    mapped_pt = np.where(idx >= 0, pt[idx], np.nan)
    mapped_pb = np.where(idx >= 0, pb[idx], np.nan)
    mapped_dt = np.where(idx >= 0, dt[idx], np.nan)
    mapped_db = np.where(idx >= 0, db[idx], np.nan)
    mapped_valid = np.where(idx >= 0, valid[idx], False)

    return mapped_eq, mapped_pt, mapped_pb, mapped_dt, mapped_db, mapped_valid


# ---------------------------------------------------------------------------
# SMA Trend Strength Oscillator Filter (MQL5 port)
#
# Ported from SMATrendStrengthOscillator.mq5 v1.01. That version added
# ENUM_MA_METHOD_TYPE (SMA or EMA, applied to BOTH the fast and slow MA) --
# the previous .mq5 was SMA-only, and this port was SMA-only to match.
# `ma_method` mirrors that new input: "SMA" (default, old behavior) or "EMA".
# Everything downstream of the two MAs (osc/signal/range/color) is unchanged
# and was already verified to match the .mq5 loop-based averaging exactly.
# ---------------------------------------------------------------------------

def _compute_ma(close: pd.Series, period: int, ma_method: str) -> pd.Series:
    if ma_method == "EMA":
        # MQL5 iMA(..., MODE_EMA, ...) seeds the EMA with an SMA of the
        # first `period` closes, then applies the standard EMA recursion
        # from there -- this matches pandas' adjust=False EMA once you
        # give it that same seed via min_periods, so we replicate MQL5's
        # seeding explicitly rather than relying on pandas' own warm-up,
        # which uses a different (adjust=True-style) initial weighting.
        alpha = 2.0 / (period + 1.0)
        ema = pd.Series(np.nan, index=close.index)
        if len(close) >= period:
            seed = close.iloc[:period].mean()
            ema.iloc[period - 1] = seed
            prev = seed
            for i in range(period, len(close)):
                prev = close.iloc[i] * alpha + prev * (1.0 - alpha)
                ema.iloc[i] = prev
        return ema
    # default / "SMA"
    return close.rolling(period).mean()


def compute_sma_trend_filter(df, fast_period, slow_period, signal_period,
                             range_period, mult, ma_method="SMA"):
    close = df["close"]
    fast_sma = _compute_ma(close, fast_period, ma_method)
    slow_sma = _compute_ma(close, slow_period, ma_method)

    # MQL5 guards against slow==0.0 or slow==EMPTY_VALUE by forcing
    # OscBuffer[i]=0.0 and OscColors[i]=2 at that bar (see .mq5 OnCalculate),
    # rather than letting the division blow up. Mirror that here so any bar
    # where slow_sma is exactly 0 or NaN never produces inf/NaN downstream.
    invalid_slow = slow_sma.isna() | (slow_sma == 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        osc = ((fast_sma - slow_sma) / slow_sma) * 100.0
    osc = osc.where(~invalid_slow, 0.0)

    first_start = slow_period + range_period
    osc_valid = osc.copy()
    osc_valid.iloc[:first_start] = np.nan

    signal = osc_valid.rolling(signal_period, min_periods=1).mean()

    abs_osc = osc_valid.abs()
    range_level = abs_osc.rolling(range_period, min_periods=1).mean() * mult

    color = np.full(len(df), 2, dtype=int)
    trending = (abs_osc >= range_level) & range_level.notna()
    color[trending & (osc >= 0)] = 0
    color[trending & (osc < 0)] = 1
    # Force color=2 (range) on invalid-slow bars, matching .mq5's explicit
    # OscColors[i]=2 fallback -- overrides whatever the comparison above gave.
    color[invalid_slow.to_numpy()] = 2

    return osc, signal, range_level, color


# ---------------------------------------------------------------------------
# Dynamic Market Structure Index (DMSI) – exact MQL5 port
#
# Ports DynamicMarketStructureIndex.mq5 v2.10's OnCalculate math bar-for-bar.
# MQL5 arrays are series (index 0 = newest bar); this port works in plain
# chronological order (index 0 = oldest bar), so every MQL5 buffer access
# `WC(i + j)` (older-by-j-bars than the newest computed bar i) becomes
# `close[c - j]` (older-by-j-bars than chronological bar c).
#
# Two details that are easy to get wrong when porting this and were fixed
# after a careful re-read of the .mq5 source:
#
#  1. The ER window and the regression window are NOT the same width.
#     net_change/gross_change touch WC(i) .. WC(i+n) inclusive -- that's
#     n+1 closes (n differences). The regression loop touches WC(i) ..
#     WC(i+n-1) inclusive -- only n closes. So ER's "n" bars-back reach is
#     one bar further than the regression's.
#
#  2. Because ER reaches to WC(i+n), the true data requirement for a given
#     bar is `i+n <= rates_total-1`, which is what MQL5's
#     `max_calc_idx = rates_total - InpLookback - 1` actually encodes. In
#     chronological terms this means the FIRST fully-computable bar is at
#     chronological index `lookback` (not `lookback - 1`); everything
#     before that is warm-up (default ExtHistColorBuffer = 2.0 / Range).
#
#  - Efficiency Ratio (ER): net_change over an (n+1)-wide window / sum of
#    the n bar-to-bar absolute changes in that window.
#  - Linear-regression R^2 over the n-wide window WC(i)..WC(i+n-1). MQL5's
#    x=j runs 0..n-1 with j=0 at the newest bar in the window (x increases
#    going backward in time); R^2 is unaffected by this orientation since
#    it only depends on slope*sum_xy and sum_y2/sum_y, which are invariant
#    to reversing both x and y order together. Slope/intercept are not
#    used by the filter (only the MQL5 "Direction" plot uses them, which
#    this port omits since it doesn't feed the regime gate), so the port
#    below computes R^2 in plain oldest->newest order for clarity -- same
#    numeric result, verified against MQL5's j-ordering by hand.
#  - Weighted fusion: raw = (er*er_weight + r2*(1-er_weight)) * 100
#  - Adaptive EMA smoothing: sc = (er*(fastest-slowest) + slowest) ** power
#  - Rolling percentile bands (InpPercentilePeriod) for Trend/Range levels
#  - Regime histogram color: 0=Trend, 1=Transition, 2=Range
# ---------------------------------------------------------------------------

def compute_dmsi(df, lookback=20, percentile_period=80, trend_pct=60, range_pct=30,
                  er_weight=0.65, fast_smooth=3, slow_smooth=22, smooth_power=1.15):
    # Defaults above mirror DynamicMarketStructureIndex.mq5's own input
    # defaults (InpLookback=20, InpPercentilePeriod=80, InpTrendPercentile=60,
    # InpRangePercentile=30, InpERWeight=0.65, InpFastSmooth=3,
    # InpSlowSmooth=22, InpSmoothPower=1.15) -- previously these defaults
    # were 25/100/70/25/0.55/3/32/1.32, which matched neither the .mq5 nor
    # v2_config.py's DMSI_* values. In normal operation config.py supplies
    # every argument explicitly via getattr(...), so these are never hit --
    # but if this is ever called without config, it should fall back to the
    # .mq5's real defaults, not silently different numbers.
    close = df["close"].values.astype(float)
    n = len(close)

    # MQL5's ExtDMSIBuffer starts life filled with EMPTY_VALUE (DBL_MAX) for
    # every cell, not 0.0 -- the warm-up loop only overwrites cells above
    # max_calc_idx with 0.0 explicitly; cells at/after `first_i` are 0.0 only
    # once Pass 1 assigns them. We replicate that distinction with a sentinel
    # instead of np.zeros(), so the reset check below (`prev == 0.0 or prev
    # == EMPTY_VALUE`) is a faithful bar-for-bar port rather than a version
    # that happens to work because dmsi[c-1] is never really EMPTY_VALUE.
    EMPTY_VALUE = np.finfo(np.float64).max
    dmsi = np.full(n, EMPTY_VALUE)
    trend_level = np.full(n, np.nan)
    range_level = np.full(n, np.nan)
    hist = np.zeros(n)
    color = np.full(n, 2, dtype=int)  # default Range (fail-safe, mirrors MQL5 warm-up default)

    # MQL5: if(rates_total < InpLookback + 20) return(0);
    if n < lookback + 20:
        dmsi[dmsi == EMPTY_VALUE] = 0.0  # don't leak the internal sentinel to callers
        return dmsi, trend_level, range_level, hist, color

    # Regression constants (n-wide window: WC(i)..WC(i+n-1), i.e. lookback closes)
    nd = float(lookback)
    idx_x = np.arange(lookback, dtype=float)
    sum_x = nd * (nd - 1.0) * 0.5
    sum_x2 = (nd - 1.0) * nd * (2.0 * nd - 1.0) / 6.0
    denom = nd * sum_x2 - sum_x * sum_x
    inv_n = 1.0 / nd

    fastest = 2.0 / (fast_smooth + 1.0)
    slowest = 2.0 / (slow_smooth + 1.0)
    sc_range = fastest - slowest

    erw = max(0.55, min(0.85, er_weight))
    r2w = 1.0 - erw

    # First fully-computable chronological bar: needs `lookback` closes for
    # the regression window (indices c-lookback+1 .. c) PLUS one more bar
    # back for ER's net_change reach (index c-lookback) -- i.e. `lookback+1`
    # closes total, so the first valid c is `lookback` (0-indexed).
    first_i = lookback

    # --- Pass 1: ER, R^2, raw score, adaptive DMSI smoothing ---
    for c in range(first_i, n):
        # ER window: (lookback+1) closes, indices [c-lookback .. c] inclusive
        er_window = close[c - lookback: c + 1]
        net_change = abs(er_window[-1] - er_window[0])          # |WC(i) - WC(i+n)|
        gross_change = np.abs(np.diff(er_window)).sum()          # sum |WC(i+j)-WC(i+j+1)|
        er = 0.0
        if gross_change > 1e-12:
            er = net_change / gross_change
            if er > 1.0:
                er = 1.0

        # Regression window: lookback closes, indices [c-lookback+1 .. c] inclusive
        y = close[c - lookback + 1: c + 1]
        sum_y = y.sum()
        sum_xy = (idx_x * y).sum()
        sum_y2 = (y * y).sum()

        slope = 0.0
        intercept = 0.0
        if denom > 1e-12:
            slope = (nd * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) * inv_n

        sst = sum_y2 - sum_y * sum_y * inv_n
        sse = sum_y2 - intercept * sum_y - slope * sum_xy
        if sse < 0.0:
            sse = 0.0
        if sst < 1e-12:
            sst = 1e-12
        r2 = 1.0 - sse / sst
        r2 = 0.0 if r2 < 0.0 else (1.0 if r2 > 1.0 else r2)

        raw_scaled = (er * erw + r2 * r2w) * 100.0

        sc = er * sc_range + slowest
        sc = sc ** smooth_power

        # MQL5: reset (rather than smooth) if this is the first computable
        # bar, OR if the previous bar's DMSI value was exactly 0.0 OR
        # EMPTY_VALUE (i.e. `ExtDMSIBuffer[i+1] == 0.0 || == EMPTY_VALUE` in
        # series terms, where i+1 is the chronologically-earlier neighbor ->
        # dmsi[c-1] here). Both conditions are real branches in the source:
        # 0.0 is not just warm-up handling -- if raw_scaled ever lands
        # exactly on 0.0 mid-series, the smoothing chain restarts on the
        # next bar rather than treating 0.0 as a normal EMA anchor. The
        # EMPTY_VALUE check matters for any cell the loop reads before Pass
        # 1 has ever written to it.
        prev = dmsi[c - 1]
        if c == first_i or prev == 0.0 or prev == EMPTY_VALUE:
            dmsi[c] = raw_scaled
        else:
            dmsi[c] = dmsi[c - 1] + sc * (raw_scaled - dmsi[c - 1])

    # --- Pass 2: rolling percentile Trend/Range bands + histogram color ---
    dmsi_series = pd.Series(dmsi)
    for c in range(first_i, n):
        available = c - first_i + 1
        window = min(available, percentile_period)
        if window < 5:
            trend_level[c] = 70.0
            range_level[c] = 30.0
        else:
            w = dmsi_series.iloc[c - window + 1: c + 1].values
            sw = np.sort(w)
            idx_t = min(int(trend_pct / 100.0 * (window - 1)), window - 1)
            idx_r = min(int(range_pct / 100.0 * (window - 1)), window - 1)
            trend_level[c] = sw[idx_t]
            range_level[c] = sw[idx_r]

        mid = (trend_level[c] + range_level[c]) * 0.5
        hist[c] = dmsi[c] - mid

        if dmsi[c] >= trend_level[c]:
            color[c] = 0   # Trend
        elif dmsi[c] <= range_level[c]:
            color[c] = 2   # Range
        else:
            color[c] = 1   # Transition

    dmsi[dmsi == EMPTY_VALUE] = 0.0  # don't leak the internal sentinel to callers
    return dmsi, trend_level, range_level, hist, color


# ---------------------------------------------------------------------------
# Pattern detection – MQL5-exact Japanese patterns only
# ---------------------------------------------------------------------------

def _safe_divide(numerator, denominator):
    """Elementwise numerator/denominator, returning NaN wherever denominator
    is 0 instead of computing inf/nan and letting numpy warn about it.
    np.where(cond, a/b, fallback) still *evaluates* a/b everywhere first
    (including where b==0), which is what raises 'invalid value encountered
    in divide' even though the bad values are discarded afterward -- this
    avoids that by only dividing where it's safe."""
    out = np.full_like(numerator, np.nan, dtype=float)
    mask = denominator > 0
    np.divide(numerator, denominator, out=out, where=mask)
    return out

def _body_ratio_mask(o, h, l, c, ratio):
    r = h - l
    body = np.abs(c - o)
    br = _safe_divide(body, r)
    return np.where(r > 0, br >= ratio, False)

def _body_ratio_most_mask(o, h, l, c, ratio):
    r = h - l
    body = np.abs(c - o)
    br = _safe_divide(body, r)
    return np.where(r > 0, br <= ratio, True)

def detect_mql5_patterns(df, qualified_mask, htf_trend, require_htf_agree):
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    n = len(df)

    bear_now   = c < o
    bull_now   = c > o
    doji_shape = _body_ratio_most_mask(o, h, l, c, config.PATTERN_DOJI_BODY_RATIO)

    body_ratio = _safe_divide(np.abs(c - o), h - l)
    doji_ratio = config.PATTERN_DOJI_BODY_RATIO
    spin_ratio = config.PATTERN_SPINNING_TOP_BODY_RATIO
    spin_shape = (body_ratio > doji_ratio) & (body_ratio <= spin_ratio)

    bear_prev = np.roll(bear_now, 1); bear_prev[0] = False
    bull_prev = np.roll(bull_now, 1); bull_prev[0] = False
    o_prev = np.roll(o, 1); c_prev = np.roll(c, 1); h_prev = np.roll(h, 1); l_prev = np.roll(l, 1)

    prev_body_ratio = _safe_divide(np.abs(c_prev - o_prev), h_prev - l_prev)
    prev_long_body = prev_body_ratio >= config.PATTERN_LONG_BODY_RATIO

    piercing_mask = (
        bear_prev & prev_long_body &
        bull_now &
        (o < l_prev) &
        (c > c_prev)
    )

    dark_cloud_mask = (
        bull_prev & prev_long_body &
        bear_now &
        (o > h_prev) &
        (c < c_prev)
    )

    bull_engulf_mask = (
        bear_prev & bull_now &
        ~doji_shape & ~spin_shape &
        (o <= c_prev) & (c > o_prev) &
        (l < l_prev) & (h > h_prev)
    )

    bear_engulf_mask = (
        bull_prev & bear_now &
        ~doji_shape & ~spin_shape &
        (o >= c_prev) & (c < o_prev) &
        (h > h_prev) & (l < l_prev)
    )

    custom_piercing_mask = (
        bear_prev & bull_now &
        (l < l_prev) & (c > c_prev)
    )

    custom_dark_cloud_mask = (
        bull_prev & bear_now &
        (h > h_prev) & (c < c_prev)
    )

    # --- Hammer / Shooting Star (Nison-style, single-candle) ---
    body = np.abs(c - o)
    rng = h - l
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    wick_ratio = config.PATTERN_HAMMER_WICK_RATIO
    body_ratio_max = config.PATTERN_HAMMER_BODY_RATIO

    small_body_ratio = _safe_divide(body, rng)
    small_body = np.where(rng > 0, small_body_ratio <= body_ratio_max, False)
    have_body = body > 0

    hammer_mask = (
        small_body & have_body &
        (lower_wick >= body * wick_ratio) &
        (upper_wick <= body * 0.5)
    )

    shooting_star_mask = (
        small_body & have_body &
        (upper_wick >= body * wick_ratio) &
        (lower_wick <= body * 0.5)
    )

    # --- MQL5 v1.04-exact priority: SINGLE chain, one label max per candle.
    # BullishEngulfing/BearishEngulfing still short-circuit the whole bar
    # (MQL5's EvaluatePatternsAt returns immediately on an engulfing hit).
    # Everything else -- Piercing, DarkCloudCover, CustomPiercing,
    # CustomDarkCloud, Hammer, ShootingStar -- is now a single if/elif-style
    # priority chain across BOTH directions, matching the corrected .mq5
    # (v1.04) EvaluatePatternsAt: at most ONE pattern total can fire on a
    # given candle, never one bull tag + one bear tag simultaneously.
    # Priority order (first match wins), identical to the .mq5 fix:
    #   Piercing > DarkCloudCover > CustomPiercing > CustomDarkCloud >
    #   Hammer > ShootingStar
    code = np.zeros(n, dtype=int)
    code[bull_engulf_mask] = 3
    code[bear_engulf_mask] = 4

    open_mask = code == 0
    code[piercing_mask & open_mask] = 1
    open_mask = code == 0

    code[dark_cloud_mask & open_mask] = 2
    open_mask = code == 0

    code[custom_piercing_mask & open_mask] = 5
    open_mask = code == 0

    code[custom_dark_cloud_mask & open_mask] = 6
    open_mask = code == 0

    code[hammer_mask & open_mask] = 7
    open_mask = code == 0

    code[shooting_star_mask & open_mask] = 8

    code_info = {
        1: ("Piercing", True, 2),
        2: ("DarkCloudCover", False, 2),
        3: ("BullishEngulfing", True, 2),
        4: ("BearishEngulfing", False, 2),
        5: ("CustomPiercing", True, 2),
        6: ("CustomDarkCloud", False, 2),
        7: ("Hammer", True, 1),
        8: ("ShootingStar", False, 1),
    }

    hits = []
    times = df["time"]

    # At most one code per bar now, so this is a single pass -- no more
    # combining separate bull/bear code arrays.
    codes_per_bar = [(i, code[i]) for i in np.where((code > 0) & qualified_mask)[0]]

    if config.DEBUG_PATTERN_SKIPS:
        total_detected = int((code > 0).sum())
        print(f"[Debug] Total pattern hits detected: {total_detected}")

    for i, code in codes_per_bar:
        name, is_bull, span = code_info[code]
        if require_htf_agree:
            lo = max(0, i - span + 1)
            wanted_trend = 1 if is_bull else -1
            align = np.any(htf_trend[lo:i+1] == wanted_trend)
            if not align:
                if config.DEBUG_PATTERN_SKIPS:
                    print(f"[Debug] Skipped {name} at {times.iloc[i]} (no HTF agreement)")
                continue
        pattern_low  = np.min(l[i-span+1:i+1])
        pattern_high = np.max(h[i-span+1:i+1])
        sl_extreme = pattern_low if is_bull else pattern_high
        hits.append({
            "time": times.iloc[i].strftime("%Y-%m-%d %H:%M:%S"),
            "bar_index": int(i),
            "pattern": name,
            "direction": "bull" if is_bull else "bear",
            "open": float(o[i]), "high": float(h[i]),
            "low": float(l[i]), "close": float(c[i]),
            "sl_extreme": sl_extreme,
            "library": "japanese",
        })

    if config.DEBUG_PATTERN_SKIPS:
        print(f"[Debug] Patterns after HTF agreement: {len(hits)}")

    return hits


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

def get_buffered_sl(entry, direction, sl_extreme, buffer_pct):
    """Return stop-loss widened by buffer_pct % of entry price."""
    if direction == "bull":
        return sl_extreme - entry * (buffer_pct / 100.0)
    else:
        return sl_extreme + entry * (buffer_pct / 100.0)


def _resolve_conflict_on_ltf(ltf_slice, entry, risk, is_bull, use_breakeven,
                              be_trigger_price, be_stop, current_sl, tp,
                              use_open_sl, breakeven_armed_in):
    """Replay a single CTF bar at 1min resolution to find the true
    chronological first hit among BE-arm/BE-stop/SL/TP. Mirrors the exact
    same state machine as the CTF-bar loop in simulate_trade, just stepped
    minute-by-minute instead of once per CTF bar. Returns a dict describing
    the first terminal event ("breakeven"/"sl"/"tp") plus updated
    breakeven_armed state, or None if nothing resolves within this slice
    (caller falls back to whatever the CTF-bar-level check would have done).
    """
    breakeven_armed = breakeven_armed_in
    lo = ltf_slice["low"].to_numpy()
    hi = ltf_slice["high"].to_numpy()
    op = ltf_slice["open"].to_numpy()
    cl = ltf_slice["close"].to_numpy()

    for j in range(len(ltf_slice)):
        m_open = float(op[j]); m_high = float(hi[j])
        m_low = float(lo[j]); m_close = float(cl[j])

        armed_entering_candle = breakeven_armed
        arm_reachable = use_breakeven and not armed_entering_candle and (
            (m_high >= be_trigger_price) if is_bull else (m_low <= be_trigger_price)
        )

        # Within THIS single 1min candle, arming, BE-stop, SL, and/or TP can
        # still collide the same way CTF bars did -- a 1min bar only gives
        # open/high/low/close too, so if this candle's own range reaches
        # more than one of them we still can't read the order off the OHLC
        # directly. Resolve with a directional distance-from-open heuristic
        # (see _event_order below): not certainty, but far better than a
        # fixed BE > SL > TP priority order that ignores the candle's own
        # shape entirely. Critically, if BE only arms *within* this same
        # candle (not already armed entering it), its stop-out can only be
        # considered reachable if the arm-trigger level is ordered before
        # the stop level -- a trade can't stop out on a trigger it hasn't
        # armed yet.
        be_stop_reachable_raw = (armed_entering_candle or arm_reachable) and (
            (m_low <= be_stop) if is_bull else (m_high >= be_stop)
        )
        sl_reachable = (m_low <= current_sl) if is_bull else (m_high >= current_sl)
        tp_reachable = (m_high >= tp) if is_bull else (m_low <= tp)

        def _event_order():
            """Order the reachable events (including the BE arm-trigger
            itself, when arming would happen in this same candle) by a
            directional distance-from-open heuristic.

            arm/be_stop/tp all sit on the SAME (favorable) side of price as
            each other -- arm and tp are beyond open in the favorable
            direction, be_stop sits between entry and open-ish territory.
            Ordering these three by raw |level - open| is fine since
            they're directly comparable along one path. But sl sits on the
            OPPOSITE side (unfavorable), reached via a separate wick that
            has no time-ordering relationship to the favorable-side wick
            implied by proximity-to-open on the other side. Comparing sl's
            raw distance from open against arm's raw distance from open
            tells you nothing about which happened first -- a candle can
            wick both ways in either order. So: rank favorable-side events
            (arm, be_stop, tp) by their own proximity-to-open among
            themselves, rank sl on its own (trivially first among
            unfavorable-side events since it's the only one), then merge
            the two sides by which wick is more extreme relative to open --
            i.e. whichever side's furthest reachable level is farther from
            open is treated as the side the candle's larger wick visited,
            and assumed to have happened first (larger excursions plausibly
            reflect the earlier, more violent part of the candle's range
            before it settled toward close). This removes the old bias
            where 'arm' (often the farthest favorable-side level) was
            silently penalized by being compared on the same axis as sl
            regardless of side.
            """
            favorable_side = []
            if be_stop_reachable_raw:
                favorable_side.append(("breakeven", be_stop))
            if tp_reachable:
                favorable_side.append(("tp", tp))
            if arm_reachable:
                favorable_side.append(("arm", be_trigger_price))
            favorable_side.sort(key=lambda ev: abs(ev[1] - m_open))

            unfavorable_side = []
            if sl_reachable:
                unfavorable_side.append(("sl", current_sl))

            if not favorable_side:
                return unfavorable_side
            if not unfavorable_side:
                return favorable_side

            fav_extreme_dist = abs(favorable_side[-1][1] - m_open)
            unfav_extreme_dist = abs(unfavorable_side[-1][1] - m_open)
            if unfav_extreme_dist > fav_extreme_dist:
                return unfavorable_side + favorable_side
            return favorable_side + unfavorable_side

        ordered = _event_order()

        # If arming happens in this same candle, BE-stop can only actually
        # fire if "arm" precedes "breakeven" in the resolved order --
        # otherwise the stop level was reached before the trigger armed it,
        # so it doesn't count this candle (but the candle's high/low may
        # still arm it for the NEXT candle onward).
        be_stop_reachable = be_stop_reachable_raw
        if arm_reachable and be_stop_reachable_raw:
            arm_pos = next(i for i, ev in enumerate(ordered) if ev[0] == "arm")
            be_pos = next(i for i, ev in enumerate(ordered) if ev[0] == "breakeven")
            be_stop_reachable = arm_pos < be_pos

        if arm_reachable:
            breakeven_armed = True

        candidates = [ev for ev in ordered if ev[0] in ("sl", "tp") or
                      (ev[0] == "breakeven" and be_stop_reachable)]

        # Walk candidates in resolved order rather than only inspecting the
        # first one. This matters for SL under use_open_sl: SL sorting
        # first doesn't mean the trade actually closes there -- it only
        # realizes on a close beyond the level. If it doesn't confirm, the
        # next candidate in line (which may be "breakeven" OR "tp", not
        # just "tp") is still live and must be checked, not silently
        # dropped.
        for ev_name, _ in candidates:
            if ev_name == "breakeven":
                exit_price = be_stop
                r = (exit_price - entry) / risk if is_bull else (entry - exit_price) / risk
                return {"outcome": "breakeven", "r_multiple": r, "exit_price": exit_price,
                        "breakeven_armed": True, "ltf_index": j}

            if ev_name == "sl":
                if use_open_sl:
                    sl_hit = (m_close <= current_sl) if is_bull else (m_close >= current_sl)
                    if sl_hit:
                        exit_price = m_close
                        r = (exit_price - entry) / risk if is_bull else (entry - exit_price) / risk
                        return {"outcome": "sl", "r_multiple": r, "exit_price": exit_price,
                                "breakeven_armed": breakeven_armed, "ltf_index": j}
                    # use_open_sl means SL only actually realizes on a close
                    # beyond the level -- the intrabar touch alone doesn't
                    # close the trade. Don't return here: fall through to
                    # the next candidate in `candidates` (breakeven or tp,
                    # whichever is next in resolved order) instead of
                    # special-casing just one of them.
                    continue
                else:
                    exit_price = current_sl
                    r = (exit_price - entry) / risk if is_bull else (entry - exit_price) / risk
                    return {"outcome": "sl", "r_multiple": r, "exit_price": exit_price,
                            "breakeven_armed": breakeven_armed, "ltf_index": j}

            if ev_name == "tp":
                return {"outcome": "tp", "r_multiple": None, "exit_price": tp,
                        "breakeven_armed": breakeven_armed, "ltf_index": j}

    return {"outcome": None, "breakeven_armed": breakeven_armed}


def simulate_trade(hit, o, h, l, c, n, rr, use_breakeven, be_trigger, be_buffer,
                   overlapping, use_open_sl, ltf_df=None, ctf_times=None, ctf_delta=None):
    entry_idx = hit["bar_index"]
    entry = float(c[entry_idx])
    is_bull = hit["direction"] == "bull"
    sl_extreme = hit["sl_extreme"]
    sl = get_buffered_sl(entry, "bull" if is_bull else "bear", sl_extreme, config.SL_BUFFER_PCT)
    risk = abs(entry - sl)
    if risk <= 0:
        return {
            "outcome": "invalid",
            "r_multiple": 0.0,
            "max_favorable_r": 0.0,
            "max_favorable_pct": 0.0,
            "max_favorable_r_b4_sl": None,
            "entry": entry,
            "sl": sl,
            "tp": None,
            "exit_price": None,
            "exit_time": None,
            "breakeven_hit": False,
        }

    tp_source = "rr"
    if getattr(config, "TP_MODE", "RR") == "PERCENT":
        tp_pct = config.TP_PERCENT / 100.0
        pct_tp = entry * (1 + tp_pct) if is_bull else entry * (1 - tp_pct)

        max_r = getattr(config, "TP_PERCENT_MAX_R", None)
        if max_r is not None:
            max_r_tp = entry + max_r * risk if is_bull else entry - max_r * risk
            # Whichever target sits closer to entry is reached first as price
            # moves in the favorable direction, so that's the effective TP --
            # i.e. the trade closes on whichever of the two levels is hit first.
            if is_bull:
                tp = min(pct_tp, max_r_tp)
            else:
                tp = max(pct_tp, max_r_tp)
            tp_source = "max_r" if tp == max_r_tp else "percent"
        else:
            tp = pct_tp
            tp_source = "percent"
    else:
        tp = entry + rr * risk if is_bull else entry - rr * risk

    tp_r_multiple = (tp - entry) / risk if is_bull else (entry - tp) / risk

    be_mode = getattr(config, "BE_MODE", "RR")
    be_stop = entry + be_buffer * risk if is_bull else entry - be_buffer * risk
    if be_mode == "PERCENT":
        be_trigger_pct = config.BE_TRIGGER_PERCENT / 100.0
        be_trigger_price = entry * (1 + be_trigger_pct) if is_bull else entry * (1 - be_trigger_pct)
    else:
        be_trigger_price = entry + be_trigger * risk if is_bull else entry - be_trigger * risk

    current_sl = sl
    breakeven_armed = False
    max_favorable_r = 0.0
    max_favorable_price = entry

    have_ltf = ltf_df is not None and len(ltf_df) > 0 and ctf_times is not None and ctf_delta is not None
    ltf_time_vals = ltf_df["time"].to_numpy() if have_ltf else None

    def _fav_pct():
        return ((max_favorable_price - entry) / entry * 100.0) if is_bull \
            else ((entry - max_favorable_price) / entry * 100.0)

    for idx in range(entry_idx + 1, n):
        bar_high = float(h[idx])
        bar_low = float(l[idx])

        fav_extreme = bar_high if is_bull else bar_low
        bar_fav_r = (fav_extreme - entry) / risk if is_bull else (entry - fav_extreme) / risk
        if bar_fav_r > max_favorable_r:
            max_favorable_r = bar_fav_r
        if is_bull:
            if fav_extreme > max_favorable_price:
                max_favorable_price = fav_extreme
        else:
            if fav_extreme < max_favorable_price:
                max_favorable_price = fav_extreme

        # Would BE-arm-then-stop, SL, and/or TP each be triggerable purely
        # from this CTF bar's own high/low? If more than one could fire,
        # the CTF bar alone can't tell us which happened first -- that's
        # the conflict. Drop to 1min data for this bar's window to find
        # the true chronological order.
        would_be_stop = use_breakeven and (
            breakeven_armed or (
                (is_bull and bar_high >= be_trigger_price) or
                (not is_bull and bar_low <= be_trigger_price)
            )
        ) and ((bar_low <= be_stop) if is_bull else (bar_high >= be_stop))
        would_sl = (bar_low <= current_sl) if is_bull else (bar_high >= current_sl)
        would_tp = (bar_high >= tp) if is_bull else (bar_low <= tp)
        n_candidates = sum([would_be_stop, would_sl, would_tp])

        if n_candidates > 1 and have_ltf:
            bar_start = ctf_times[idx]
            bar_end = bar_start + ctf_delta
            mask = (ltf_time_vals >= bar_start) & (ltf_time_vals < bar_end)
            ltf_slice = ltf_df[mask]
            if len(ltf_slice) > 0:
                res = _resolve_conflict_on_ltf(
                    ltf_slice, entry, risk, is_bull, use_breakeven,
                    be_trigger_price, be_stop, current_sl, tp, use_open_sl,
                    breakeven_armed,
                )
                breakeven_armed = res["breakeven_armed"]
                if res["outcome"] == "breakeven":
                    r = res["r_multiple"]
                    return {
                        "outcome": "breakeven",
                        "r_multiple": round(r, 4),
                        "max_favorable_r": round(max_favorable_r, 4),
                        "max_favorable_pct": round(_fav_pct(), 4),
                        "max_favorable_r_b4_sl": None,
                        "entry": entry, "sl": sl, "tp": tp,
                        "exit_price": res["exit_price"],
                        "exit_index": idx,
                        "breakeven_hit": True,
                        "conflict_resolved_ltf": True,
                    }
                if res["outcome"] == "sl":
                    r = res["r_multiple"]
                    return {
                        "outcome": "sl",
                        "r_multiple": round(r, 4),
                        "max_favorable_r": round(max_favorable_r, 4),
                        "max_favorable_pct": round(_fav_pct(), 4),
                        "max_favorable_r_b4_sl": None,
                        "entry": entry, "sl": sl, "tp": tp,
                        "exit_price": res["exit_price"],
                        "exit_index": idx,
                        "breakeven_hit": False,
                        "conflict_resolved_ltf": True,
                    }
                if res["outcome"] == "tp":
                    return {
                        "outcome": "tp",
                        "r_multiple": round(tp_r_multiple, 4),
                        "max_favorable_r": round(max_favorable_r, 4),
                        "max_favorable_pct": round(_fav_pct(), 4),
                        "max_favorable_r_b4_sl": None,
                        "entry": entry, "sl": sl, "tp": tp,
                        "tp_source": tp_source,
                        "exit_price": tp,
                        "exit_index": idx,
                        "breakeven_hit": False,
                        "conflict_resolved_ltf": True,
                    }
                # res["outcome"] is None (LTF slice didn't resolve it, e.g.
                # partial/missing minutes) -- fall through to normal
                # CTF-bar-level logic below using original breakeven_armed
                # from before the LTF attempt (arming itself is safe to
                # keep since would_be_stop already re-derives it there too).

        # NOTE: this block only runs when the LTF conflict-resolver above
        # didn't handle the bar (no LTF data available, or the LTF slice
        # was empty/inapplicable for this bar). It must NOT blindly return
        # "breakeven" the instant BE-stop is reachable -- if arming
        # happens on this SAME bar (not already armed entering it), TP
        # may also be reachable on this bar, and without LTF data to
        # arbitrate we have no reliable evidence BE-stop was actually
        # touched *before* TP. In that ambiguous case, prefer TP: it is
        # the trade's actual target and assuming the worst-case ordering
        # (BE before TP) systematically undercounts wins whenever the
        # entry/arming candle is a strong impulse straight to target.
        armed_entering_bar = breakeven_armed

        if use_breakeven and not breakeven_armed:
            if is_bull and bar_high >= be_trigger_price:
                breakeven_armed = True
            elif not is_bull and bar_low <= be_trigger_price:
                breakeven_armed = True

        be_stop_hit = breakeven_armed and (
            (bar_low <= be_stop) if is_bull else (bar_high >= be_stop)
        )
        tp_hit = (bar_high >= tp) if is_bull else (bar_low <= tp)
        armed_this_bar = breakeven_armed and not armed_entering_bar

        if be_stop_hit and tp_hit and armed_this_bar:
            # Same-bar arm + BE-stop + TP, and we have no LTF data to
            # arbitrate the true order -- resolve in favor of TP rather
            # than defaulting to BE, since forcing BE here on ambiguous
            # same-bar evidence would silently convert real wins into
            # breakevens whenever the entry candle is a strong impulse.
            return {
                "outcome": "tp",
                "r_multiple": round(tp_r_multiple, 4),
                "max_favorable_r": round(max_favorable_r, 4),
                "max_favorable_pct": round(_fav_pct(), 4),
                "max_favorable_r_b4_sl": None,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "tp_source": tp_source,
                "exit_price": tp,
                "exit_index": idx,
                "breakeven_hit": False,
                "conflict_resolved_ltf": False,
                "same_bar_arm_be_tp_conflict": True,
            }

        if be_stop_hit:
            exit_price = be_stop
            r = (exit_price - entry) / risk if is_bull else (entry - exit_price) / risk
            return {
                "outcome": "breakeven",
                "r_multiple": round(r, 4),
                "max_favorable_r": round(max_favorable_r, 4),
                "max_favorable_pct": round(_fav_pct(), 4),
                "max_favorable_r_b4_sl": None,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit_price": exit_price,
                "exit_index": idx,
                "breakeven_hit": True,
            }

        if use_open_sl:
            sl_hit = (c[idx] <= current_sl) if is_bull else (c[idx] >= current_sl)
        else:
            sl_hit = (bar_low <= current_sl) if is_bull else (bar_high >= current_sl)

        if sl_hit:
            exit_price = c[idx] if use_open_sl else current_sl
            r = (exit_price - entry) / risk if is_bull else (entry - exit_price) / risk
            return {
                "outcome": "sl",
                "r_multiple": round(r, 4),
                "max_favorable_r": round(max_favorable_r, 4),
                "max_favorable_pct": round(_fav_pct(), 4),
                "max_favorable_r_b4_sl": None,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit_price": exit_price,
                "exit_index": idx,
                "breakeven_hit": False,
            }

        if tp_hit:
            return {
                "outcome": "tp",
                "r_multiple": round(tp_r_multiple, 4),
                "max_favorable_r": round(max_favorable_r, 4),
                "max_favorable_pct": round(_fav_pct(), 4),
                "max_favorable_r_b4_sl": None,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "tp_source": tp_source,
                "exit_price": tp,
                "exit_index": idx,
                "breakeven_hit": False,
            }

    return {
        "outcome": "open",
        "r_multiple": None,
        "max_favorable_r": round(max_favorable_r, 4),
        "max_favorable_pct": round(_fav_pct(), 4),
        "max_favorable_r_b4_sl": None,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "exit_price": None,
        "exit_index": None,
        "breakeven_hit": breakeven_armed,
    }


def compute_original_sl_max_fav(hit, o, h, l, c, n):
    entry_idx = hit["bar_index"]
    entry = float(c[entry_idx])
    is_bull = hit["direction"] == "bull"
    sl_extreme = hit["sl_extreme"]
    sl = get_buffered_sl(entry, "bull" if is_bull else "bear", sl_extreme, config.SL_BUFFER_PCT)
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0

    max_fav_r = 0.0
    for idx in range(entry_idx + 1, n):
        bar_high = float(h[idx]); bar_low = float(l[idx])
        fav_extreme = bar_high if is_bull else bar_low
        fav_r = (fav_extreme - entry) / risk if is_bull else (entry - fav_extreme) / risk
        if fav_r > max_fav_r:
            max_fav_r = fav_r
        if is_bull and bar_low <= sl:
            break
        if not is_bull and bar_high >= sl:
            break
    return max_fav_r


# ---------------------------------------------------------------------------
# Reporting functions
# ---------------------------------------------------------------------------

BOLD = "\033[1m"; GREEN = "\033[32m"; RED = "\033[31m"
YELLOW = "\033[33m"; CYAN = "\033[36m"; RESET = "\033[0m"
BLUE = "\033[34m"
BG_HIGHLIGHT = "\033[1;37;44m"  # bold white text on blue background
MAX_FAV_R_HIGHLIGHT_THRESHOLD = 9.0

def box_line(line: str, width: int = 100):
    print(f"│ {line[:width-2]:<{width-2}} │")

def print_detailed_trades(hits, ctf_df, ltf_df=None):
    n = len(ctf_df)
    o = ctf_df["open"].to_numpy()
    h = ctf_df["high"].to_numpy()
    l = ctf_df["low"].to_numpy()
    c = ctf_df["close"].to_numpy()
    ctf_times = ctf_df["time"].to_numpy()
    ctf_delta = (ctf_df["time"].iloc[1] - ctf_df["time"].iloc[0]) if n > 1 else None

    overlapping = config.OVERLAPPING_TRADES
    blocked_until_idx = -1

    for hit in hits:
        if not overlapping and hit["bar_index"] < blocked_until_idx:
            hit["trade"] = {
                "outcome": "skipped_overlap", "r_multiple": None,
                "max_favorable_r": None, "max_favorable_pct": None, "max_favorable_r_b4_sl": None,
                "entry": None, "sl": None, "tp": None,
                "exit_price": None, "exit_time": None,
                "breakeven_hit": False,
            }
            continue

        trade = simulate_trade(
            hit, o, h, l, c, n,
            rr=config.RR,
            use_breakeven=config.USE_BREAKEVEN,
            be_trigger=config.BREAKEVEN_TRIGGER,
            be_buffer=config.BREAKEVEN_BUFFER,
            overlapping=overlapping,
            use_open_sl=config.USE_OPEN_SL,
            ltf_df=ltf_df,
            ctf_times=ctf_times,
            ctf_delta=ctf_delta,
        )

        trade["max_favorable_r_b4_sl"] = compute_original_sl_max_fav(
            hit, o, h, l, c, n
        )

        if not overlapping:
            exit_idx = trade.get("exit_index")
            blocked_until_idx = (exit_idx + 1) if exit_idx is not None else n

        if trade.get("exit_index") is not None:
            trade["exit_time"] = ctf_df["time"].iloc[trade["exit_index"]].strftime("%Y-%m-%d %H:%M:%S")
            del trade["exit_index"]
        if config.USE_COMMISSION and trade.get("outcome") in ("tp","sl","breakeven"):
            trade["r_multiple_gross"] = trade["r_multiple"]
            trade["r_multiple"] = round(trade["r_multiple"] - config.COMMISSION_R, 4)
        hit["trade"] = trade

    W_SN = 5; W_TIME = 19; W_PATTERN = 15; W_ENTRY = 13
    W_OUTCOME = 5; W_OUTCOME_R = 6; W_MAXPCT = 9; W_MFE = 8; W_MFEb4 = 12; W_CUM = 10

    show_outcome_r = getattr(config, "SHOW_OUTCOME_R", True)

    if show_outcome_r:
        header_fields = [
            f"{'S/N':<{W_SN}}", f"{'DateTime':<{W_TIME}}", f"{'Pattern':<{W_PATTERN}}",
            f"{'Entry':>{W_ENTRY}}", f"{'Out':<{W_OUTCOME}}",
            f"{'Out R':>{W_OUTCOME_R}}", f"{'Max (%)':>{W_MAXPCT}}", f"{'MaxFavR':>{W_MFE}}", f"{'MaxFavR_b4SL':>{W_MFEb4}}",
            f"{'Cum R':>{W_CUM}}",
        ]
    else:
        header_fields = [
            f"{'S/N':<{W_SN}}", f"{'DateTime':<{W_TIME}}", f"{'Pattern':<{W_PATTERN}}",
            f"{'Entry':>{W_ENTRY}}", f"{'Out':<{W_OUTCOME}}",
            f"{'Max (%)':>{W_MAXPCT}}", f"{'MaxFavR':>{W_MFE}}", f"{'MaxFavR_b4SL':>{W_MFEb4}}", f"{'Cum R':>{W_CUM}}",
        ]
    inner = " ".join(header_fields)
    box_width = len(inner) + 2

    hdr = "│ " + inner + " │"
    sep = "│ " + "─"*len(inner) + " │"

    print("\n┌" + "─"*box_width + "┐")
    title = " DETAILED TRADE LOG (all simulated trades)"
    print("│" + title + " "*(box_width - len(title)) + "│")
    print("├" + "─"*box_width + "┤")
    print(hdr)
    print(sep)

    net_r = 0.0
    for sn, hit in enumerate(hits, 1):
        t = hit["trade"]
        entry = t.get("entry")
        entry_str = f"{entry:>{W_ENTRY}.5f}" if entry is not None else f"{'--':>{W_ENTRY}}"
        outcome = t.get("outcome", "?")
        r_val = t.get("r_multiple")
        mfe = t.get("max_favorable_r")
        mfe_plain = f"{mfe:+.2f}R" if mfe is not None else "  --"
        mfe_str = (f"{BG_HIGHLIGHT}{mfe_plain:>{W_MFE}}{RESET}"
                   if mfe is not None and mfe >= MAX_FAV_R_HIGHLIGHT_THRESHOLD
                   else f"{mfe_plain:>{W_MFE}}")
        max_pct = t.get("max_favorable_pct")
        max_pct_plain = f"{max_pct:+.2f}%" if max_pct is not None else "  --"
        max_pct_str = (f"{BG_HIGHLIGHT}{max_pct_plain:>{W_MAXPCT}}{RESET}"
                       if mfe is not None and mfe >= MAX_FAV_R_HIGHLIGHT_THRESHOLD
                       else f"{max_pct_plain:>{W_MAXPCT}}")
        mfe_b4 = t.get("max_favorable_r_b4_sl")
        mfe_b4_plain = f"{mfe_b4:+.2f}R" if mfe_b4 is not None else "  --"
        mfe_b4_str = (f"{BG_HIGHLIGHT}{mfe_b4_plain:>{W_MFEb4}}{RESET}"
                      if mfe_b4 is not None and mfe_b4 >= MAX_FAV_R_HIGHLIGHT_THRESHOLD
                      else f"{mfe_b4_plain:>{W_MFEb4}}")
        if r_val is not None:
            net_r += r_val
        cum_str = f"{net_r:+.2f}R"

        pat_color = GREEN if hit["direction"] == "bull" else RED
        pat_colored = f"{pat_color}{hit['pattern'][:W_PATTERN]:<{W_PATTERN}}{RESET}"

        display_outcome = outcome
        if outcome == "breakeven": display_outcome = "BE"
        elif outcome == "skipped_overlap": display_outcome = "SKIP"

        if outcome == "tp": out_color = GREEN
        elif outcome == "breakeven": out_color = BLUE
        elif outcome == "sl": out_color = RED
        elif outcome == "skipped_overlap": out_color = YELLOW
        else: out_color = RESET
        out_colored = f"{out_color}{display_outcome[:W_OUTCOME]:<{W_OUTCOME}}{RESET}"

        if r_val is not None:
            if outcome == "breakeven":
                out_r_color = BLUE
            else:
                out_r_color = GREEN if r_val > 0 else (RED if r_val < 0 else RESET)
            out_r_str = f"{r_val:+.2f}R"
        else:
            out_r_color = RESET; out_r_str = "  --"
        out_r_colored = f"{out_r_color}{out_r_str:>{W_OUTCOME_R}}{RESET}"

        cum_color = GREEN if net_r > 0 else (RED if net_r < 0 else RESET)
        cum_colored = f"{cum_color}{cum_str:>{W_CUM}}{RESET}"

        if show_outcome_r:
            row = (f"│ {sn:<{W_SN}} {hit['time']:<{W_TIME}} {pat_colored} "
                   f"{entry_str} {out_colored} {out_r_colored} {max_pct_str} {mfe_str} "
                   f"{mfe_b4_str} {cum_colored} │")
        else:
            row = (f"│ {sn:<{W_SN}} {hit['time']:<{W_TIME}} {pat_colored} "
                   f"{entry_str} {out_colored} {max_pct_str} {mfe_str} "
                   f"{mfe_b4_str} {cum_colored} │")
        print(row)
        print("│" + " "*box_width + "│")
    print("└" + "─"*box_width + "┘")
    return net_r


def ascii_sparkline(values, width=98):
    if not values or width <= 0: return ""
    levels = " ▁▂▃▄▅▆▇█"
    n = len(values)
    if n >= width:
        sampled = [values[int(round(i * (n - 1) / (width - 1)))] for i in range(width)]
    else:
        sampled = [values[int(i * n / width)] for i in range(width)]
    lo, hi = min(sampled), max(sampled)
    span = hi - lo
    chars = []
    for v in sampled:
        if span <= 0:
            level = len(levels) // 2
        else:
            level = 1 + int(round((v - lo) / span * (len(levels) - 2)))
            level = max(1, min(len(levels) - 1, level))
        chars.append(levels[level])
    return "".join(chars)


def ascii_equity_chart(values, width=98, height=12):
    if not values or width <= 0 or height <= 0:
        return ["[no data]"]

    n = len(values)
    if n >= width:
        sampled = [values[int(round(i * (n - 1) / (width - 1)))] for i in range(width)]
    else:
        sampled = [values[int(i * n / width)] for i in range(width)]

    lo, hi = min(sampled + [0.0]), max(sampled + [0.0])
    span = hi - lo
    if span <= 0:
        span = 1.0

    def row_for(v):
        frac = (v - lo) / span
        r = int(round((height - 1) * (1 - frac)))
        return max(0, min(height - 1, r))

    zero_row = row_for(0.0)
    point_rows = [row_for(v) for v in sampled]

    axis_w = 9
    grid = [[" "] * width for _ in range(height)]

    for x in range(width):
        grid[zero_row][x] = "-"

    prev_row = point_rows[0]
    grid[prev_row][0] = "*"
    for x in range(1, width):
        cur_row = point_rows[x]
        lo_r, hi_r = min(prev_row, cur_row), max(prev_row, cur_row)
        for r in range(lo_r, hi_r + 1):
            grid[r][x] = "*"
        prev_row = cur_row

    for x in range(width):
        if grid[zero_row][x] == " ":
            grid[zero_row][x] = "-"

    lines = []
    for r in range(height):
        if r == 0:
            label = f"{hi:+.1f}R"
        elif r == height - 1:
            label = f"{lo:+.1f}R"
        elif r == zero_row:
            label = "0.0R"
        else:
            label = ""
        label_padded = label.rjust(axis_w - 1) + "|" if label else " " * (axis_w - 1) + "|"
        lines.append(label_padded + "".join(grid[r]))

    lines.append(" " * axis_w + "└" + "─" * width)
    return lines


# ANSI color codes for the equity curve health check block. Degrades to
# plain text automatically if the terminal doesn't render ANSI, since the
# escape codes just print as harmless invisible/no-op sequences on most
# modern terminals -- but callers can still hard-disable via
# config.USE_ANSI_COLOR = False if needed.
_ANSI_GREEN  = "\033[92m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RED    = "\033[91m"
_ANSI_BOLD   = "\033[1m"
_ANSI_RESET  = "\033[0m"

def _color_enabled():
    return getattr(config, "USE_ANSI_COLOR", True)

def _colorize(text, color):
    if not _color_enabled():
        return text
    return f"{color}{text}{_ANSI_RESET}"

def _verdict_color(good, warn):
    """good/warn are booleans; good=True -> green, warn=True -> yellow,
    else red."""
    if good:
        return _ANSI_GREEN
    if warn:
        return _ANSI_YELLOW
    return _ANSI_RED

def box_line_colored(line: str, visible_len: int, width: int = 100):
    """Like box_line, but `line` may already contain ANSI escape codes.
    `visible_len` is the length of the line as it will actually appear
    on screen (i.e. len(line) with escape codes stripped out), so padding
    lines up correctly even though the raw string is longer than that."""
    pad = max(0, (width - 2) - visible_len)
    print(f"│ {line}{' ' * pad} │")


def print_equity_curve_health(equity_curve, r_vals, dd, trailing_dd):
    """Evaluate the equity curve against the standard 'healthy equity curve'
    criteria (steady upward slope, shallow drawdowns, fast recoveries, low
    trade-to-trade volatility, no reliance on a couple of outlier wins) and
    print a verdict block. Uses data already computed in
    print_performance_stats -- no re-derivation of trade outcomes.
    Printed as the LAST block of the report."""
    if len(equity_curve) < 2 or not r_vals:
        print("\n┌" + "─"*100 + "┐")
        box_line(_colorize("EQUITY CURVE HEALTH CHECK", _ANSI_BOLD) if _color_enabled() else "EQUITY CURVE HEALTH CHECK")
        print("├" + "─"*100 + "┤")
        box_line("Not enough closed trades to assess.")
        print("└" + "─"*100 + "┘")
        return

    final_r = equity_curve[-1]

    # --- Slope / consistency: fraction of trade-to-trade steps that are up
    steps = [equity_curve[i] - equity_curve[i-1] for i in range(1, len(equity_curve))]
    up_steps = sum(1 for s in steps if s > 0)
    slope_consistency = up_steps / len(steps) * 100 if steps else 0.0

    # --- Drawdown depth relative to the peak achieved (proxy for % of
    #     "account equity" in R terms, since this is an R-based curve)
    peak = max(equity_curve) if equity_curve else 0.0
    dd_pct_of_peak = (dd / peak * 100) if peak > 0 else (100.0 if dd > 0 else 0.0)

    # --- Recovery speed: trades needed to reach a new equity high after
    #     each drawdown-from-peak episode, averaged
    running_peak = -float("inf")
    in_drawdown = False
    recovery_lengths = []
    trades_since_peak = 0
    for v in equity_curve:
        if v >= running_peak:
            if in_drawdown:
                recovery_lengths.append(trades_since_peak)
            running_peak = v
            in_drawdown = False
            trades_since_peak = 0
        else:
            in_drawdown = True
            trades_since_peak += 1
    avg_recovery = np.mean(recovery_lengths) if recovery_lengths else 0.0

    # --- Volatility between trades: coefficient of variation of R outcomes
    avg_r = np.mean(r_vals)
    std_r = np.std(r_vals, ddof=1) if len(r_vals) > 1 else 0.0

    # --- Outlier dependence: share of total positive R coming from the
    #     single/two largest winning trades
    wins_sorted = sorted([r for r in r_vals if r > 0], reverse=True)
    total_wins_r = sum(wins_sorted)
    top1_share = (wins_sorted[0] / total_wins_r * 100) if wins_sorted and total_wins_r > 0 else 0.0
    top2_share = (sum(wins_sorted[:2]) / total_wins_r * 100) if len(wins_sorted) >= 2 and total_wins_r > 0 else top1_share

    header_text = "EQUITY CURVE HEALTH CHECK"
    print("\n┌" + "─"*100 + "┐")
    box_line_colored(_colorize(header_text, _ANSI_BOLD), len(header_text))
    print("├" + "─"*100 + "┤")
    box_line("A healthy curve climbs steadily with shallow pullbacks, quick recoveries, low")
    box_line("trade-to-trade volatility, and growth spread across many trades rather than a")
    box_line("couple of outsized wins.")
    box_line("─" * 96)

    # 1. Steady upward slope
    if slope_consistency >= 55 and final_r > 0:
        slope_verdict = "STEADY"
    elif final_r > 0:
        slope_verdict = "CHOPPY"
    else:
        slope_verdict = "DOWNWARD"
    slope_color = _verdict_color(slope_verdict == "STEADY", slope_verdict == "CHOPPY")
    label = f"Slope:              {slope_consistency:.1f}% of trades net-positive step  ->  ["
    tag = f"{slope_verdict}]"
    box_line_colored(label + _colorize(tag, slope_color), len(label) + len(tag))

    # 2. Shallow drawdowns (rule of thumb: <15-20% of peak equity)
    if dd_pct_of_peak <= 20:
        dd_verdict = "SHALLOW"
    elif dd_pct_of_peak <= 40:
        dd_verdict = "MODERATE"
    else:
        dd_verdict = "DEEP"
    dd_color = _verdict_color(dd_verdict == "SHALLOW", dd_verdict == "MODERATE")
    label = f"Drawdown depth:     {dd:.3f}R max DD  ({dd_pct_of_peak:.1f}% of peak {peak:.3f}R)  ->  ["
    tag = f"{dd_verdict}]"
    box_line_colored(label + _colorize(tag, dd_color), len(label) + len(tag))

    # 3. Fast recoveries
    if not recovery_lengths:
        rec_verdict = "N/A"
        rec_note = "(no drawdown episodes)"
    elif avg_recovery <= 3:
        rec_verdict = "FAST"
        rec_note = ""
    elif avg_recovery <= 8:
        rec_verdict = "MODERATE"
        rec_note = ""
    else:
        rec_verdict = "SLOW"
        rec_note = ""
    rec_color = _verdict_color(rec_verdict == "FAST", rec_verdict in ("MODERATE", "N/A"))
    if recovery_lengths:
        label = (f"Recovery speed:     avg {avg_recovery:.1f} trades to new high across "
                  f"{len(recovery_lengths)} drawdown episode(s)  ->  [")
    else:
        label = f"Recovery speed:     {rec_note}  ->  ["
    tag = f"{rec_verdict}]"
    box_line_colored(label + _colorize(tag, rec_color), len(label) + len(tag))

    # 4. Low trade-to-trade volatility
    cv = (std_r / abs(avg_r)) if avg_r != 0 else float("inf")
    if cv <= 1.5:
        vol_verdict = "LOW"
    elif cv <= 3.0:
        vol_verdict = "MODERATE"
    else:
        vol_verdict = "HIGH"
    vol_color = _verdict_color(vol_verdict == "LOW", vol_verdict == "MODERATE")
    label = f"Trade volatility:   mean {avg_r:+.3f}R, std {std_r:.3f}R (CV {cv:.2f})  ->  ["
    tag = f"{vol_verdict}]"
    box_line_colored(label + _colorize(tag, vol_color), len(label) + len(tag))

    # 5. Outlier dependence
    if top1_share <= 20:
        outlier_verdict = "WELL DISTRIBUTED"
    elif top1_share <= 40:
        outlier_verdict = "SOME CONCENTRATION"
    else:
        outlier_verdict = "OUTLIER-DEPENDENT"
    outlier_color = _verdict_color(outlier_verdict == "WELL DISTRIBUTED", outlier_verdict == "SOME CONCENTRATION")
    label = f"Win concentration:  top win {top1_share:.1f}% of gross profit, top 2 = {top2_share:.1f}%  ->  ["
    tag = f"{outlier_verdict}]"
    box_line_colored(label + _colorize(tag, outlier_color), len(label) + len(tag))

    box_line("─" * 96)

    flags = sum([
        slope_verdict != "STEADY",
        dd_verdict == "DEEP",
        rec_verdict == "SLOW",
        vol_verdict == "HIGH",
        outlier_verdict == "OUTLIER-DEPENDENT",
    ])
    if flags == 0:
        overall = "HEALTHY -- consistent with strict risk rules and steady execution."
        overall_color = _ANSI_GREEN
    elif flags <= 2:
        overall = "MIXED -- broadly reasonable, but one or two traits need attention."
        overall_color = _ANSI_YELLOW
    else:
        overall = "UNHEALTHY -- multiple warning signs (choppiness, deep DD, slow recovery,"
        overall_color = _ANSI_RED
    label = "Overall: ["
    tag = f"{overall}]" if flags == 0 or flags <= 2 else overall
    full_label = label
    box_line_colored(_colorize(full_label, _ANSI_BOLD) + _colorize(tag, overall_color),
                      len(full_label) + len(tag))
    if flags > 2:
        extra = "high variance, or outlier reliance)"
        box_line_colored(_colorize(extra, overall_color), len(extra))
    print("└" + "─"*100 + "┘")


def print_performance_stats(hits, net_r):
    if not hits:
        print("\n[Performance] No trades found. Cannot compute performance statistics.")
        return {
            "n_closed": 0, "n_open": 0, "n_wins": 0, "n_losses": 0,
            "win_rate": 0.0, "total_r": 0.0, "expectancy": 0.0,
            "n_tp": 0, "n_sl": 0, "n_be": 0, "max_dd": 0.0,
            "worst_month_dd": 0.0,
        }

    closed = [h["trade"] for h in hits if h["trade"]["outcome"] not in ("open","invalid","skipped_overlap")]
    outcomes = [t["outcome"] for t in closed]
    r_vals = [t["r_multiple"] for t in closed if t["r_multiple"] is not None]

    n_tp = outcomes.count("tp")
    n_sl = outcomes.count("sl")
    n_be = outcomes.count("breakeven")
    n_skipped = sum(1 for h in hits if h["trade"]["outcome"] == "skipped_overlap")
    n_open = len(hits) - len(closed) - n_skipped
    n_closed = len(closed)

    n_wins = sum(1 for r in r_vals if r > 0)
    n_losses = n_closed - n_wins
    win_rate = (n_wins / n_closed * 100) if n_closed else 0.0

    total_r = sum(r_vals)
    expectancy = total_r / len(r_vals) if r_vals else 0.0

    print("\n" + "="*70)
    print("Global trade statistics:")
    print("-"*70)
    skip_note = f"   (Skipped-overlap: {n_skipped})" if not config.OVERLAPPING_TRADES else ""
    print(f"  Total closed trades: {n_closed}   (Open: {n_open}){skip_note}")
    print(f"  Outcomes -> TP: {n_tp}   SL: {n_sl}   Breakeven: {n_be}")
    print(f"  Wins: {n_wins}   Losses: {n_losses}")
    print(f"  Win rate: {win_rate:.2f}%")
    print(f"  Total R: {total_r:+.2f}   Expectancy: {expectancy:+.3f}R")
    print("-"*70)

    # Monthly stats
    months = {}
    for hit in hits:
        t = hit["trade"]
        if t["outcome"] in ("open","invalid","skipped_overlap"): continue
        m = hit["time"][:7]
        months.setdefault(m, {"trades": 0, "wins": 0, "be": 0, "losses": 0, "r": 0.0, "peak": 0.0, "dd": 0.0})
        months[m]["trades"] += 1
        if t["outcome"] == "breakeven":
            months[m]["be"] += 1
        elif t.get("r_multiple", 0) > 0:
            months[m]["wins"] += 1
        else:
            months[m]["losses"] += 1
        months[m]["r"] += t["r_multiple"]

    month_trade_seq = []
    for hit in hits:
        t = hit["trade"]
        if t["outcome"] in ("open","invalid","skipped_overlap"): continue
        r = t.get("r_multiple", 0)
        m = hit["time"][:7]
        month_trade_seq.append((m, r))

    cur_month = None; cum_in_month = 0.0; peak_in_month = 0.0; dd_in_month = 0.0
    for m, r in month_trade_seq:
        if m != cur_month:
            if cur_month is not None:
                months[cur_month]["peak"] = peak_in_month; months[cur_month]["dd"] = dd_in_month
            cur_month = m; cum_in_month = 0.0; peak_in_month = 0.0; dd_in_month = 0.0
        cum_in_month += r
        if cum_in_month > peak_in_month: peak_in_month = cum_in_month
        draw_from_peak = peak_in_month - cum_in_month
        if draw_from_peak > dd_in_month: dd_in_month = draw_from_peak
    if cur_month is not None:
        months[cur_month]["peak"] = peak_in_month; months[cur_month]["dd"] = dd_in_month

    # Daily stats
    days = {}
    for hit in hits:
        t = hit["trade"]
        if t["outcome"] in ("open","invalid","skipped_overlap"): continue
        d = hit["time"][:10]
        days.setdefault(d, {"trades": 0, "wins": 0, "losses": 0, "r": 0.0, "peak": 0.0, "dd": 0.0})
        day = days[d]
        day["trades"] += 1
        r = t.get("r_multiple", 0)
        if r > 0: day["wins"] += 1
        else: day["losses"] += 1
        day["r"] += r

    trade_seq = []
    for hit in hits:
        t = hit["trade"]
        if t["outcome"] in ("open","invalid","skipped_overlap"): continue
        r = t.get("r_multiple", 0)
        d = hit["time"][:10]
        trade_seq.append((d, r))

    cur_day = None; cum_in_day = 0.0; peak_in_day = 0.0; dd_in_day = 0.0
    for d, r in trade_seq:
        if d != cur_day:
            if cur_day is not None:
                days[cur_day]["peak"] = peak_in_day; days[cur_day]["dd"] = dd_in_day
            cur_day = d; cum_in_day = 0.0; peak_in_day = 0.0; dd_in_day = 0.0
        cum_in_day += r
        if cum_in_day > peak_in_day: peak_in_day = cum_in_day
        draw_from_peak = peak_in_day - cum_in_day
        if draw_from_peak > dd_in_day: dd_in_day = draw_from_peak
    if cur_day is not None:
        days[cur_day]["peak"] = peak_in_day; days[cur_day]["dd"] = dd_in_day

    cum_r = 0.0; peak = 0.0; dd = 0.0
    equity_curve = [0.0]
    for hit in hits:
        t = hit["trade"]
        if t["outcome"] in ("open","invalid","skipped_overlap"): continue
        r = t.get("r_multiple", 0)
        cum_r += r; equity_curve.append(cum_r)
        if cum_r > peak: peak = cum_r
        draw = peak - cum_r
        if draw > dd: dd = draw

    running_peak = 0.0
    for v in equity_curve:
        if v > running_peak: running_peak = v
    trailing_dd = running_peak - equity_curve[-1] if equity_curve else 0.0

    print("\n┌" + "─"*100 + "┐")
    box_line("RISK-ADJUSTED PERFORMANCE")
    print("├" + "─"*100 + "┤")
    start = hits[0]["time"][:10] if hits else "N/A"
    end = hits[-1]["time"][:10] if hits else "N/A"
    days_count = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days + 1
    years = days_count / 365.25
    if years > 0:
        cagr = (np.exp(np.log(1 + total_r / 100) / years) - 1) * 100 if total_r > -100 else -100.0
    else:
        cagr = 0.0
    mar = cagr / abs(dd * 100) if dd > 0 else 0
    box_line(f"Period: {days_count} days (~{years:.2f}yr)   Net R: {total_r:+.3f}")
    box_line(f"Max Net R: {peak:+.3f}   Min Net R: {cum_r - peak:+.3f}")
    if len(equity_curve) > 1:
        chart_width = 88
        chart_height = 14
        box_line(f"Equity curve (R), {len(equity_curve)-1} trades:")
        for line in ascii_equity_chart(equity_curve, width=chart_width, height=chart_height):
            box_line(line)
    else:
        box_line("Equity curve (R)  [no closed trades]")
    print("├" + "─"*100 + "┤")
    if len(r_vals) > 1:
        avg_r = np.mean(r_vals); std_r = np.std(r_vals, ddof=1)
        sharpe = (avg_r / std_r) * np.sqrt(252) if std_r else 0
        downside = [r for r in r_vals if r < 0]
        sortino = (avg_r / (np.std(downside, ddof=1) or 1e-6)) * np.sqrt(252) if downside else np.inf
    else:
        sharpe = sortino = 0
    box_line(f"Sharpe  (annualized, rf=0.0%): {sharpe:.3f}")
    box_line(f"Sortino (annualized, rf=0.0%): {sortino:.3f}")
    box_line(f"CAGR (R-equity curve):            {cagr:.2f}%")
    box_line(f"Max Drawdown (since inception):   {dd:.4f}R")
    box_line(f"Trailing Drawdown (current, from peak): {trailing_dd:.4f}R")
    box_line(f"MAR Ratio (CAGR / inception DD):   {mar:.3f}")
    print("├" + "─"*100 + "┤")
    box_line("                            CALMAR RATIO")
    calmar = mar
    if calmar < 0:
        box_line(f"  Calmar = {calmar:.2f}   [POOR (losing strategy)]")
    elif calmar < 1:
        box_line(f"  Calmar = {calmar:.2f}   [WEAK]")
    elif calmar < 2:
        box_line(f"  Calmar = {calmar:.2f}   [ACCEPTABLE]")
    elif calmar < 3:
        box_line(f"  Calmar = {calmar:.2f}   [GOOD]")
    elif calmar < 5:
        box_line(f"  Calmar = {calmar:.2f}   [EXCELLENT]")
    else:
        box_line(f"  Calmar = {calmar:.2f}   [EXCEPTIONAL]")
    print("└" + "─"*100 + "┘")

    if getattr(config, "SHOW_DAILY_BREAKDOWN", False):
        print("\n┌" + "─"*100 + "┐")
        box_line("DAILY BREAKDOWN (peak → drawdown sequence)")
        print("├" + "─"*100 + "┤")
        box_line("Date          Trades   Wins   Losses   WinRate     Day R     Peak R     Day DD       Cum R")
        print("│ " + "─"*97 + "│")
        cum = 0.0
        for d in sorted(days):
            day = days[d]
            wr = day["wins"] / day["trades"] * 100 if day["trades"] else 0
            cum += day["r"]
            box_line(f"{d:<10} {day['trades']:>7} {day['wins']:>6} {day['losses']:>8} {wr:>7.2f}% {day['r']:>10.2f}R {day['peak']:>10.2f}R {day['dd']:>10.2f}R {cum:>10.2f}R")
        print("└" + "─"*100 + "┘")

    print("\n┌" + "─"*100 + "┐")
    box_line("MONTH-BY-MONTH BREAKDOWN")
    print("├" + "─"*100 + "┤")
    box_line("Month       Trades   Wins     BE   Losses   WinRate     Month R      Max DD       Cum R")
    print("│ " + "─"*97 + "│")
    cum = 0.0
    for m in sorted(months):
        mo = months[m]
        wr = mo["wins"] / mo["trades"] * 100 if mo["trades"] else 0
        cum += mo["r"]
        box_line(f"{m:<7} {mo['trades']:>7} {mo['wins']:>6} {mo['be']:>6} {mo['losses']:>8} {wr:>7.2f}% {mo['r']:>11.2f}R {mo['dd']:>11.2f}R {cum:>11.2f}R")
    print("└" + "─"*100 + "┘")

    print("\n" + "-"*70)
    print("Per‑pattern trade statistics:")
    print("-"*70)
    pattern_stats = {}
    for hit in hits:
        pat = hit["pattern"]
        t = hit["trade"]
        if pat not in pattern_stats:
            pattern_stats[pat] = {"tp":0,"sl":0,"be":0,"open":0,"r_sum":0.0,"wins":0,"count":0}
        ps = pattern_stats[pat]
        if t["outcome"] in ("tp","sl","breakeven"):
            stat_key = "be" if t["outcome"] == "breakeven" else t["outcome"]
            ps[stat_key] = ps.get(stat_key, 0) + 1
            ps["count"] += 1
            if t["r_multiple"] is not None:
                ps["r_sum"] += t["r_multiple"]
                if t["r_multiple"] > 0: ps["wins"] += 1
        elif t["outcome"] == "open":
            ps["open"] += 1
    for pat, ps in pattern_stats.items():
        wr = ps["wins"] / ps["count"] * 100 if ps["count"] else 0
        print(f"Pattern: {pat}")
        print(f"  TP: {ps['tp']}   SL: {ps['sl']}   BE: {ps['be']}   Open: {ps['open']}")
        if ps["count"]:
            print(f"  Win rate: {wr:.2f}%   Total R: {ps['r_sum']:+.2f}   Expectancy: {ps['r_sum']/ps['count']:+.3f}R")
        else:
            print("  No closed trades")
    print("-"*70)

    worst_month_dd = max((mo["dd"] for mo in months.values()), default=0.0)

    print_equity_curve_health(equity_curve, r_vals, dd, trailing_dd)

    return {
        "n_closed": n_closed, "n_open": n_open, "n_wins": n_wins, "n_losses": n_losses,
        "win_rate": win_rate, "total_r": total_r, "expectancy": expectancy,
        "n_tp": n_tp, "n_sl": n_sl, "n_be": n_be, "max_dd": dd,
        "worst_month_dd": worst_month_dd,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scan(ctf_timeframe, is_multi=False, symbol=None):
    """Run the full scan + trade simulation for a single (symbol, CTF
    timeframe) pair. Mutates config.CTF_TIMEFRAME / config.SYMBOL /
    config.CSV_PATH / config.SL_BUFFER_PCT for the duration of the call
    (every helper below reads them from `config`), then returns a summary
    dict for cross-run comparison. `symbol=None` keeps whatever
    config.SYMBOL already is (backward-compatible single-symbol behavior)."""
    config.CTF_TIMEFRAME = ctf_timeframe
    if symbol is not None:
        config.SYMBOL = symbol
        config.CSV_PATH = config.csv_path_for(symbol)
        config.SL_BUFFER_PCT = config.sl_buffer_pct_for(symbol)

    print(f"[Scanner] symbol={config.SYMBOL}  ctf_tf={config.CTF_TIMEFRAME}  "
          f"sl_buffer_pct={config.SL_BUFFER_PCT}")
    print(f"[Scanner] loading CSV: {config.CSV_PATH}")

    df = pd.read_csv(config.CSV_PATH, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next((c for c in ("time","date","datetime") if c in df.columns), None)
    if time_col is None: raise ValueError("No time column")
    required = ["open","high","low","close"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns={time_col: "time"})
    df = df[["time","open","high","low","close"]].copy()

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_rows = df[df[["open", "high", "low", "close"]].isna().any(axis=1)]
    if len(bad_rows) > 0:
        print(f"[Scanner] WARNING: {len(bad_rows)} rows had non-numeric OHLC values, dropping them:")
        print(bad_rows.head(20))
        df = df.dropna(subset=["open", "high", "low", "close"])

    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    native_delta = detect_csv_timeframe(df)
    target_delta = pd.Timedelta(config.CTF_TIMEFRAME)

    # Preserve the native-resolution data (pre-resample) so the trade
    # simulator can drop to it for intrabar BE/SL/TP conflict resolution.
    # Only usable as a "lower timeframe" if it's actually finer than the CTF.
    ltf_df = df[["time", "open", "high", "low", "close"]].copy() if native_delta < target_delta else None

    if native_delta == target_delta:
        print(f"[CTF] CSV timeframe matches target {config.CTF_TIMEFRAME}, no resample needed")
    else:
        print(f"[CTF] detected {format_timeframe(native_delta)} -> resampling to {config.CTF_TIMEFRAME}")
        df = resample_to_tf(df, config.CTF_TIMEFRAME)

    # Time window filtering
    warmup_bars = getattr(config, "WARMUP_BARS", 0)
    trailing_bars = getattr(config, "TRAILING_BUFFER_BARS", 0)
    use_full_warmup = isinstance(warmup_bars, str) and warmup_bars.strip().upper() == "ALL"
    if config.START_DATE:
        start_ts = pd.Timestamp(config.START_DATE)
        if use_full_warmup:
            pass
        elif isinstance(warmup_bars, (int, float)) and warmup_bars > 0:
            pre_mask = df["time"] < start_ts
            n_pre = int(pre_mask.sum())
            start_row = max(0, n_pre - int(warmup_bars))
            df = df.iloc[start_row:]
        else:
            df = df[df["time"] >= start_ts]
    report_end_ts = None
    if config.END_DATE:
        end_ts = pd.Timestamp(config.END_DATE)
        if end_ts.time() == pd.Timestamp("00:00:00").time():
            end_ts += pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        report_end_ts = end_ts
        if trailing_bars > 0:
            in_window = df[df["time"] <= end_ts]
            post = df[df["time"] > end_ts].iloc[:trailing_bars]
            df = pd.concat([in_window, post], ignore_index=True)
        else:
            df = df[df["time"] <= end_ts]
    df = df.reset_index(drop=True)

    # Clip the preserved lower-timeframe data down to the same time span as
    # the (now filtered) CTF frame, so we're not carrying the whole CSV
    # around. A little headroom on the end covers the last CTF bar's full
    # width even if its close sits right at df's final timestamp.
    if ltf_df is not None and len(df) > 0:
        ltf_start = df["time"].iloc[0]
        ltf_end = df["time"].iloc[-1] + target_delta
        ltf_df = ltf_df[(ltf_df["time"] >= ltf_start) & (ltf_df["time"] < ltf_end)].reset_index(drop=True)

    have_warmup = use_full_warmup or (isinstance(warmup_bars, (int, float)) and warmup_bars > 0)
    if config.START_DATE and have_warmup:
        report_start_idx = int((df["time"] < pd.Timestamp(config.START_DATE)).sum())
    else:
        report_start_idx = 0

    print(f"[Scanner] {len(df)} CTF bars loaded ({df['time'].iloc[0]} -> {df['time'].iloc[-1]}), "
          f"{report_start_idx} warm-up bars"
          f"{' (ALL available)' if use_full_warmup else ''}, "
          f"{trailing_bars if (config.END_DATE and trailing_bars>0) else 0} trailing buffer bars after")

    avg_price = df["close"].mean()
    point = 0.01 if avg_price > 100 else (0.001 if avg_price > 10 else 0.0001)
    print(f"[Pip] point size = {point}")

    # Trend HTF and zone HTF
    if config.USE_HTF_MAPPING:
        trend_htf_tf = map_timeframe(config.CTF_TIMEFRAME)
    else:
        trend_htf_tf = config.CTF_TIMEFRAME

    if config.INP_USE_HTF_ZONES:
        zone_htf_tf = map_timeframe(config.CTF_TIMEFRAME)
    else:
        zone_htf_tf = config.CTF_TIMEFRAME

    print(f"[Indicator] Trend HTF: {trend_htf_tf}   Zones HTF: {zone_htf_tf}")

    htf_df = resample_to_tf(df, trend_htf_tf)
    if len(htf_df) > 0:
        htf_delta = pd.Timedelta(htf_df["time"].diff().mode().iloc[0])
        if htf_df["time"].iloc[-1] + htf_delta > df["time"].iloc[-1]:
            htf_df = htf_df.iloc[:-1]
    print(f"[HTF Trend] {len(htf_df)} bars ({trend_htf_tf})")

    htf_trend = compute_trend(htf_df, config.TREND_ENGINE, config.PIVOT_LENGTH,
                              config.REQUIRED_PIVOT_PAIRS, config.HOLD_LAST_TREND_ON_UNDEFINED,
                              getattr(config, "CHOCH_BOS_MIN_PIVOTS", 1))

    if config.INP_USE_HTF_ZONES:
        zone_htf_df = resample_to_tf(df, zone_htf_tf)
        if len(zone_htf_df) > 0:
            zone_delta = pd.Timedelta(zone_htf_df["time"].diff().mode().iloc[0])
            if zone_htf_df["time"].iloc[-1] + zone_delta > df["time"].iloc[-1]:
                zone_htf_df = zone_htf_df.iloc[:-1]
        print(f"[HTF Zones] {len(zone_htf_df)} bars ({zone_htf_tf})")

        htf_atr = precompute_atr(zone_htf_df, config.INP_ATR_PERIOD)
        (eq_htf, pt_htf, pb_htf, dt_htf, db_htf, valid_htf,
         buf_disc_htf, buf_prem_htf) = compute_pd_zones_vectorized(
            zone_htf_df, config.INP_LOOKBACK_PERIOD, config.INP_USE_ATR,
            config.INP_ATR_PERIOD, config.INP_ATR_MULTIPLIER,
            config.INP_BUFFER_POINTS, config.INP_USE_MANUAL_ZONES,
            config.INP_DISCOUNT_INNER_OFFSET, config.INP_DISCOUNT_OUTER_OFFSET,
            config.INP_PREMIUM_INNER_OFFSET, config.INP_PREMIUM_OUTER_OFFSET,
            point, htf_atr)
        eq, pt, pb, dt, db, zone_valid = map_htf_zones_to_ctf(zone_htf_df, df,
                                                              (eq_htf, pt_htf, pb_htf,
                                                               dt_htf, db_htf, valid_htf))
        # Buffer-distance arrays (used only by the confluence-signal port)
        # are HTF-native series -- map them onto the CTF bar grid the same
        # way the zone price levels above are mapped, via a matching-length
        # dummy 6-tuple reusing map_htf_zones_to_ctf's index alignment.
        _, buf_disc, buf_prem, _, _, _ = map_htf_zones_to_ctf(
            zone_htf_df, df,
            (buf_disc_htf, buf_prem_htf, buf_disc_htf, buf_prem_htf, buf_disc_htf,
             np.ones(len(zone_htf_df), dtype=bool)))
    else:
        print("[Indicator] computing zones on CTF")
        ctf_atr = precompute_atr(df, config.INP_ATR_PERIOD)
        eq, pt, pb, dt, db, zone_valid, buf_disc, buf_prem = compute_pd_zones_vectorized(
            df, config.INP_LOOKBACK_PERIOD, config.INP_USE_ATR,
            config.INP_ATR_PERIOD, config.INP_ATR_MULTIPLIER,
            config.INP_BUFFER_POINTS, config.INP_USE_MANUAL_ZONES,
            config.INP_DISCOUNT_INNER_OFFSET, config.INP_DISCOUNT_OUTER_OFFSET,
            config.INP_PREMIUM_INNER_OFFSET, config.INP_PREMIUM_OUTER_OFFSET,
            point, ctf_atr)

    htf_times = htf_df["time"].values
    ctf_times = df["time"].values
    valid_htf_indices = np.where(~np.isnan(htf_trend))[0]
    if len(valid_htf_indices) == 0:
        mapped_htf_trend = np.zeros(len(df), dtype=int)
    else:
        last_valid_htf = valid_htf_indices[-1]
        idx_map = np.searchsorted(htf_times[:last_valid_htf+1], ctf_times, side='right') - 1
        idx_map = np.clip(idx_map, 0, last_valid_htf)
        mapped_htf_trend = np.where(idx_map >= 0, htf_trend[idx_map], 0)
        mapped_htf_trend = np.where(np.isnan(mapped_htf_trend), 0, mapped_htf_trend).astype(int)

    if config.FILTER_HTF_ALIGNMENT:
        ctf_trend = compute_trend(df, config.TREND_ENGINE, config.PIVOT_LENGTH,
                                  config.REQUIRED_PIVOT_PAIRS, config.HOLD_LAST_TREND_ON_UNDEFINED,
                                  getattr(config, "CHOCH_BOS_MIN_PIVOTS", 1))
        ctf_trend_clean = np.where(np.isnan(ctf_trend), 0, ctf_trend).astype(int)
    else:
        ctf_trend_clean = np.zeros(len(df), dtype=int)

    if config.INP_REQUIRE_ZONE_TOUCH:
        candle_high = df["high"].values
        candle_low = df["low"].values
        candle_close = df["close"].values
        candle_range = candle_high - candle_low
        overlap_threshold = max(config.INP_ZONE_OVERLAP_PERCENT, 0.0) / 100.0

        touch_ok = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            trend = mapped_htf_trend[i]
            if trend == 0:
                touch_ok[i] = False
                continue
            if trend == 1:
                zone_low = db[i]
                zone_high = dt[i]
            else:
                zone_low = pb[i]
                zone_high = pt[i]
            if np.isnan(zone_low) or np.isnan(zone_high) or zone_high <= zone_low:
                touch_ok[i] = False
                continue
            if candle_close[i] >= zone_low and candle_close[i] <= zone_high:
                touch_ok[i] = True
            elif candle_range[i] > 0:
                overlap = max(0.0, min(candle_high[i], zone_high) - max(candle_low[i], zone_low))
                if overlap >= overlap_threshold * candle_range[i]:
                    touch_ok[i] = True
    else:
        touch_ok = np.ones(len(df), dtype=bool)

    if config.USE_SMA_TREND_FILTER:
        sma_osc, sma_signal, sma_range, sma_color = compute_sma_trend_filter(
            df,
            config.SMA_FAST_PERIOD,
            config.SMA_SLOW_PERIOD,
            config.SMA_SIGNAL_PERIOD,
            config.SMA_RANGE_FILTER_PERIOD,
            config.SMA_RANGE_FILTER_MULTIPLIER,
            getattr(config, "SMA_MA_METHOD", "SMA")
        )
        print(f"[SMA Filter] enabled, colour distribution: "
              f"bull={int((sma_color == 0).sum())}, bear={int((sma_color == 1).sum())}, "
              f"range={int((sma_color == 2).sum())}")
    else:
        sma_color = np.full(len(df), 2, dtype=int)

    # Confluence-gated PD-zone Buy/Sell signals (InpShowSignals in the .mq5).
    # Off by default, matching the .mq5 default of InpShowSignals=false --
    # purely additive, does not affect qualified[]/pattern scanning unless
    # you opt in via config and consume pdz_buy_signal/pdz_sell_signal
    # yourself downstream.
    if getattr(config, "INP_SHOW_SIGNALS", False):
        pdz_buy_signal, pdz_sell_signal = compute_pd_zone_signals(
            df, eq, buf_disc, buf_prem, point,
            getattr(config, "INP_CONFLUENCE_LOOKBACK", 10),
            getattr(config, "INP_MIN_IMPULSE_POINTS", 15))
        print(f"[PD Zone Signals] enabled, buy={int(pdz_buy_signal.sum())}, "
              f"sell={int(pdz_sell_signal.sum())}")

    if getattr(config, "USE_DMSI_FILTER", False):
        dmsi_val, dmsi_trend_lvl, dmsi_range_lvl, dmsi_hist, dmsi_color = compute_dmsi(
            df,
            getattr(config, "DMSI_LOOKBACK", 20),
            getattr(config, "DMSI_PERCENTILE_PERIOD", 80),
            getattr(config, "DMSI_TREND_PERCENTILE", 60),
            getattr(config, "DMSI_RANGE_PERCENTILE", 30),
            getattr(config, "DMSI_ER_WEIGHT", 0.65),
            getattr(config, "DMSI_FAST_SMOOTH", 3),
            getattr(config, "DMSI_SLOW_SMOOTH", 22),
            getattr(config, "DMSI_SMOOTH_POWER", 1.15),
        )
        print(f"[DMSI Filter] enabled (trend-only gate), regime distribution: "
              f"trend={int((dmsi_color == 0).sum())}, transition={int((dmsi_color == 1).sum())}, "
              f"range={int((dmsi_color == 2).sum())}")
    else:
        dmsi_color = np.full(len(df), 0, dtype=int)  # neutral: never blocks when disabled

    qualified = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if mapped_htf_trend[i] == 0:
            continue
        if config.FILTER_HTF_ALIGNMENT:
            if ctf_trend_clean[i] != 0 and ctf_trend_clean[i] != mapped_htf_trend[i]:
                continue
        if config.INP_REQUIRE_ZONE_TOUCH and not touch_ok[i]:
            continue
        if config.USE_SMA_TREND_FILTER and sma_color[i] == 2:
            continue
        # DMSI gate: only take trades on a confirmed Trend (green) bar.
        # dmsi_color: 0=Trend(green), 1=Transition(yellow), 2=Range(red).
        # Warm-up bars (before the indicator has enough data) default to
        # color=2 in compute_dmsi, so they're excluded by this same check --
        # no separate "empty bar" case needed.
        if getattr(config, "USE_DMSI_FILTER", False) and dmsi_color[i] != 0:
            continue
        qualified[i] = True

    _report_mask = np.ones(len(df), dtype=bool)
    if config.START_DATE and have_warmup:
        _report_mask &= (df["time"] >= pd.Timestamp(config.START_DATE)).values
    if config.END_DATE and trailing_bars > 0 and report_end_ts is not None:
        _report_mask &= (df["time"] <= report_end_ts).values
    qualified_count = int((qualified & _report_mask).sum())
    report_bar_count = int(_report_mask.sum())
    print(f"[Indicator] qualified bars: {qualified_count} out of {report_bar_count} "
          f"(reporting window; {len(df)} total bars incl. warm-up/trailing buffer)")

    all_hits = detect_mql5_patterns(df, qualified, mapped_htf_trend, config.PATTERN_REQUIRE_HTF_AGREEMENT)

    if config.USE_SMA_TREND_FILTER and config.SMA_REQUIRE_ALIGNMENT:
        filtered_hits = []
        for h in all_hits:
            idx = h["bar_index"]
            color = sma_color[idx]
            is_bull = h["direction"] == "bull"
            if (is_bull and color == 0) or (not is_bull and color == 1):
                filtered_hits.append(h)
        all_hits = filtered_hits

    all_hits.sort(key=lambda h: h["time"])
    if config.START_DATE and have_warmup:
        all_hits = [h for h in all_hits if pd.Timestamp(h["time"]) >= pd.Timestamp(config.START_DATE)]
    if config.END_DATE and trailing_bars > 0 and report_end_ts is not None:
        all_hits = [h for h in all_hits if pd.Timestamp(h["time"]) <= report_end_ts]
    if config.PATTERN_FILTER is not None:
        all_hits = [h for h in all_hits if h["pattern"] in config.PATTERN_FILTER]
    if getattr(config, "PATTERN_EXCLUDE", None):
        all_hits = [h for h in all_hits if h["pattern"] not in config.PATTERN_EXCLUDE]

    print(f"[Indicator] pattern filter -> {config.PATTERN_FILTER or 'all'}"
          f"   exclude -> {getattr(config, 'PATTERN_EXCLUDE', None) or 'none'}")

    tp_mode = getattr(config, "TP_MODE", "RR")
    if tp_mode == "PERCENT":
        max_r = getattr(config, "TP_PERCENT_MAX_R", None)
        tp_label = f"TP={config.TP_PERCENT:g}%" + (f" (capped @ {max_r}R)" if max_r is not None else "")
    else:
        tp_label = f"RR={config.RR}"

    if config.USE_BREAKEVEN:
        be_mode = getattr(config, "BE_MODE", "RR")
        if be_mode == "PERCENT":
            be_label = f"breakeven=ON (trigger={config.BE_TRIGGER_PERCENT:g}%, buffer={config.BREAKEVEN_BUFFER}R)"
        else:
            be_label = f"breakeven=ON (trigger={config.BREAKEVEN_TRIGGER}R, buffer={config.BREAKEVEN_BUFFER}R)"
    else:
        be_label = "breakeven=OFF"

    sim_desc = [
        tp_label,
        be_label,
        f"overlapping={'ON' if config.OVERLAPPING_TRADES else 'OFF'}",
    ]
    if config.USE_OPEN_SL:
        sim_desc.append("open-SL")
    print(f"[Scanner] simulating trades: {', '.join(sim_desc)}")

    net_r = print_detailed_trades(all_hits, df, ltf_df=ltf_df)
    stats = print_performance_stats(all_hits, net_r)

    out_name = f"{config.SYMBOL}_{ctf_timeframe}.json" if is_multi else \
        (config.RESULT_FILENAME if config.RESULT_FILENAME else f"{config.SYMBOL}.json")
    out_path = os.path.join(config.RESULT_DIR, out_name)
    output = {
        "symbol": config.SYMBOL,
        "ctf_timeframe": config.CTF_TIMEFRAME,
        "sl_buffer_pct": config.SL_BUFFER_PCT,
        "trend_htf_timeframe": trend_htf_tf,
        "zone_htf_timeframe": zone_htf_tf,
        "trend_engine": config.TREND_ENGINE,
        "qualified_bars": int(qualified_count),
        "counts": {p: sum(1 for h in all_hits if h["pattern"] == p) for p in
                   ["Piercing", "DarkCloudCover", "BullishEngulfing", "BearishEngulfing",
                    "CustomPiercing", "CustomDarkCloud", "Hammer", "ShootingStar"]},
        "total_hits": len(all_hits),
        "hits": all_hits,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Scanner] JSON written to: {out_path}")

    return {
        "symbol": config.SYMBOL,
        "ctf_timeframe": ctf_timeframe,
        "qualified_bars": int(qualified_count),
        "total_hits": len(all_hits),
        "result_path": out_path,
        "hits": all_hits,
        **stats,
    }


def print_timeframe_comparison(summaries):
    print("\n┌" + "─"*100 + "┐")
    box_line("MULTI-TIMEFRAME COMPARISON")
    print("├" + "─"*100 + "┤")
    box_line(f"{'Timeframe':<10} {'Hits':>6} {'Closed':>7} {'Wins':>6} {'WinRate':>8} "
              f"{'NetR':>9} {'Exp':>8} {'MaxDD':>8}")
    print("│ " + "─"*97 + "│")
    for s in summaries:
        box_line(f"{s['ctf_timeframe']:<10} {s['total_hits']:>6} {s['n_closed']:>7} {s['n_wins']:>6} "
                  f"{s['win_rate']:>7.2f}% {s['total_r']:>+8.2f}R {s['expectancy']:>+7.3f}R {s['worst_month_dd']:>7.3f}R")

    total_hits = sum(s["total_hits"] for s in summaries)
    total_closed = sum(s["n_closed"] for s in summaries)
    total_wins = sum(s["n_wins"] for s in summaries)
    total_r_sum = sum(s["total_r"] for s in summaries)
    total_win_rate = (total_wins / total_closed * 100) if total_closed else 0.0
    total_expectancy = (total_r_sum / total_closed) if total_closed else 0.0
    total_max_dd = max((s["worst_month_dd"] for s in summaries), default=0.0)

    print("│ " + "─"*97 + "│")
    box_line(f"{'TOTAL':<10} {total_hits:>6} {total_closed:>7} {total_wins:>6} "
              f"{total_win_rate:>7.2f}% {total_r_sum:>+8.2f}R {total_expectancy:>+7.3f}R {total_max_dd:>7.3f}R")
    print("└" + "─"*100 + "┘")
    print("[Note] TOTAL row sums across timeframes -- these are resamples of the same underlying "
          "data, not independent portfolios, so NetR/WinRate here is a rollup, not a combined strategy. "
          "MaxDD is the worst single calendar-month drawdown seen in any timeframe, not the since-"
          "inception equity-curve drawdown.")
    closed_summaries = [s for s in summaries if s["n_closed"] > 0]
    if closed_summaries:
        best = max(closed_summaries, key=lambda s: s["total_r"])
        print(f"\n[Scanner] Best net R: {best['ctf_timeframe']} ({best['total_r']:+.2f}R, "
              f"win rate {best['win_rate']:.2f}%, {best['n_closed']} closed trades)")

    print("\n" + "#"*100)
    print("#  COMBINED ACROSS ALL TIMEFRAMES (pooled trades, not deduplicated)")
    print("#"*100)
    pooled_hits = []
    for s in summaries:
        pooled_hits.extend(s.get("hits", []))
    pooled_hits.sort(key=lambda h: h["time"])
    pooled_r_vals = [h["trade"]["r_multiple"] for h in pooled_hits
                      if h["trade"]["outcome"] not in ("open", "invalid", "skipped_overlap")
                      and h["trade"]["r_multiple"] is not None]
    pooled_net_r = sum(pooled_r_vals)
    print_performance_stats(pooled_hits, pooled_net_r)


def _short_symbol(symbol: str, width: int = 14) -> str:
    """Shorten a symbol name for table columns, e.g.
    'Volatility 25 (1s) Index.0_M1' -> 'Vol25(1s)'."""
    s = symbol.replace("Index.0_M1", "").replace("Index", "").strip()
    s = s.replace("Volatility ", "Vol").replace(" ", "")
    return (s[:width-1] + "…") if len(s) > width else s


def print_matrix_comparison(summaries):
    """Print a symbol x timeframe matrix of results, per-symbol and
    per-timeframe rollups, and a grand pooled-trades summary across the
    whole matrix. `summaries` is a flat list of run_scan() outputs, each
    tagged with 'symbol' and 'ctf_timeframe'."""
    symbols = list(dict.fromkeys(s["symbol"] for s in summaries))
    timeframes = list(dict.fromkeys(s["ctf_timeframe"] for s in summaries))
    grid = {(s["symbol"], s["ctf_timeframe"]): s for s in summaries}

    col_w = 12
    label_w = 16

    def fmt_cell(s):
        if s is None or s["n_closed"] == 0:
            return f"{'--':>{col_w}}"
        return f"{s['total_r']:>+{col_w-1}.2f}R"

    # --- NetR matrix -------------------------------------------------
    print("\n┌" + "─"*100 + "┐")
    box_line("SYMBOL x TIMEFRAME MATRIX -- Net R")
    print("├" + "─"*100 + "┤")
    header = f"{'Symbol':<{label_w}}" + "".join(f"{tf:>{col_w}}" for tf in timeframes) + f"{'RowSum':>{col_w}}"
    box_line(header)
    print("│ " + "─"*97 + "│")
    for sym in symbols:
        row_r = 0.0
        cells = ""
        for tf in timeframes:
            cell = grid.get((sym, tf))
            cells += fmt_cell(cell)
            if cell:
                row_r += cell["total_r"]
        box_line(f"{_short_symbol(sym):<{label_w}}" + cells + f"{row_r:>+{col_w-1}.2f}R")
    print("│ " + "─"*97 + "│")
    col_sums = ""
    grand_r = 0.0
    for tf in timeframes:
        col_r = sum(grid[(sym, tf)]["total_r"] for sym in symbols if (sym, tf) in grid)
        grand_r += col_r
        col_sums += f"{col_r:>+{col_w-1}.2f}R"
    box_line(f"{'ColSum':<{label_w}}" + col_sums + f"{grand_r:>+{col_w-1}.2f}R")
    print("└" + "─"*100 + "┘")

    # --- Win-rate matrix -----------------------------------------------
    def fmt_wr_cell(s):
        if s is None or s["n_closed"] == 0:
            return f"{'--':>{col_w}}"
        return f"{s['win_rate']:>{col_w-1}.1f}%"

    print("\n┌" + "─"*100 + "┐")
    box_line("SYMBOL x TIMEFRAME MATRIX -- Win Rate")
    print("├" + "─"*100 + "┤")
    box_line(header.replace("RowSum", "AvgWR"))
    print("│ " + "─"*97 + "│")
    for sym in symbols:
        cells = "".join(fmt_wr_cell(grid.get((sym, tf))) for tf in timeframes)
        row_closed = sum(grid[(sym, tf)]["n_closed"] for tf in timeframes if (sym, tf) in grid)
        row_wins = sum(grid[(sym, tf)]["n_wins"] for tf in timeframes if (sym, tf) in grid)
        row_wr = (row_wins / row_closed * 100) if row_closed else 0.0
        box_line(f"{_short_symbol(sym):<{label_w}}" + cells + f"{row_wr:>{col_w-1}.1f}%")
    print("└" + "─"*100 + "┘")

    # --- Per-symbol rollup ----------------------------------------------
    print("\n" + "─"*100)
    print("PER-SYMBOL ROLLUP (summed across all timeframes)")
    print("─"*100)
    print(f"{'Symbol':<{label_w}} {'Hits':>6} {'Closed':>7} {'Wins':>6} {'WinRate':>8} {'NetR':>9} {'Exp':>8} {'MaxDD':>8}")
    for sym in symbols:
        rows = [grid[(sym, tf)] for tf in timeframes if (sym, tf) in grid]
        hits = sum(r["total_hits"] for r in rows)
        closed = sum(r["n_closed"] for r in rows)
        wins = sum(r["n_wins"] for r in rows)
        net_r = sum(r["total_r"] for r in rows)
        wr = (wins / closed * 100) if closed else 0.0
        exp = (net_r / closed) if closed else 0.0
        max_dd = max((r["worst_month_dd"] for r in rows), default=0.0)
        print(f"{_short_symbol(sym):<{label_w}} {hits:>6} {closed:>7} {wins:>6} {wr:>7.2f}% "
              f"{net_r:>+8.2f}R {exp:>+7.3f}R {max_dd:>7.3f}R")

    # --- Per-timeframe rollup --------------------------------------------
    print("\n" + "─"*100)
    print("PER-TIMEFRAME ROLLUP (summed across all symbols)")
    print("─"*100)
    print(f"{'Timeframe':<{label_w}} {'Hits':>6} {'Closed':>7} {'Wins':>6} {'WinRate':>8} {'NetR':>9} {'Exp':>8} {'MaxDD':>8}")
    for tf in timeframes:
        rows = [grid[(sym, tf)] for sym in symbols if (sym, tf) in grid]
        hits = sum(r["total_hits"] for r in rows)
        closed = sum(r["n_closed"] for r in rows)
        wins = sum(r["n_wins"] for r in rows)
        net_r = sum(r["total_r"] for r in rows)
        wr = (wins / closed * 100) if closed else 0.0
        exp = (net_r / closed) if closed else 0.0
        max_dd = max((r["worst_month_dd"] for r in rows), default=0.0)
        print(f"{tf:<{label_w}} {hits:>6} {closed:>7} {wins:>6} {wr:>7.2f}% "
              f"{net_r:>+8.2f}R {exp:>+7.3f}R {max_dd:>7.3f}R")

    print("\n[Note] Rollups sum across runs on the SAME underlying price series resampled to "
          "different timeframes (per symbol) and across DIFFERENT instruments (per timeframe) -- "
          "these are rollups for comparison, not a combined portfolio backtest. MaxDD per row/column "
          "is the worst single calendar-month drawdown seen in any one run within that row/column, "
          "not a since-inception equity-curve drawdown.")

    closed_summaries = [s for s in summaries if s["n_closed"] > 0]
    if closed_summaries:
        best = max(closed_summaries, key=lambda s: s["total_r"])
        print(f"\n[Scanner] Best net R overall: {_short_symbol(best['symbol'])} @ {best['ctf_timeframe']} "
              f"({best['total_r']:+.2f}R, win rate {best['win_rate']:.2f}%, {best['n_closed']} closed trades)")

    # --- Grand pooled summary across the entire matrix -------------------
    print("\n" + "#"*100)
    print("#  COMBINED ACROSS ENTIRE SYMBOL x TIMEFRAME MATRIX (pooled trades, not deduplicated)")
    print("#"*100)
    pooled_hits = []
    for s in summaries:
        for h in s.get("hits", []):
            h = dict(h)
            h["_symbol"] = s["symbol"]
            h["_ctf_timeframe"] = s["ctf_timeframe"]
            pooled_hits.append(h)
    pooled_hits.sort(key=lambda h: h["time"])
    pooled_r_vals = [h["trade"]["r_multiple"] for h in pooled_hits
                      if h["trade"]["outcome"] not in ("open", "invalid", "skipped_overlap")
                      and h["trade"]["r_multiple"] is not None]
    pooled_net_r = sum(pooled_r_vals)
    print_performance_stats(pooled_hits, pooled_net_r)


def main():
    timeframes = config.TIMEFRAMES
    symbols = config.SYMBOLS

    is_multi_tf = len(timeframes) > 1
    is_multi_symbol = len(symbols) > 1
    is_multi = is_multi_tf or is_multi_symbol

    original_ctf = config.CTF_TIMEFRAME
    original_symbol = config.SYMBOL
    original_csv_path = config.CSV_PATH
    original_sl_buffer_pct = config.SL_BUFFER_PCT

    if os.path.exists(config.RESULT_DIR):
        shutil.rmtree(config.RESULT_DIR)
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    summaries = []
    for sym in symbols:
        for tf in timeframes:
            if is_multi:
                print("\n" + "#"*100)
                label = f"SYMBOL: {sym}  |  TIMEFRAME: {tf}" if (is_multi_symbol and is_multi_tf) \
                    else (f"SYMBOL: {sym}" if is_multi_symbol else f"TIMEFRAME: {tf}")
                print(f"#  {label}")
                print("#"*100)
            summary = run_scan(tf, is_multi=is_multi, symbol=sym)
            summaries.append(summary)

    config.CTF_TIMEFRAME = original_ctf
    config.SYMBOL = original_symbol
    config.CSV_PATH = original_csv_path
    config.SL_BUFFER_PCT = original_sl_buffer_pct

    if is_multi_symbol:
        print_matrix_comparison(summaries)
    elif is_multi_tf:
        print_timeframe_comparison(summaries)


if __name__ == "__main__":
    main()