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

import config as config


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

def compute_swing_levels(h, l, pivot_len):
    """Last *confirmed* swing high / swing low as of each bar index, using the
    same pivot_len confirmation lag as the trend-structure engines (a pivot at
    bar i is only known at bar i+pivot_len, so there's no lookahead)."""
    n = len(h)
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    if n < pivot_len * 2 + 1:
        return last_swing_high, last_swing_low

    max_eval = n - pivot_len - 1
    events_idx, events_val, events_is_high = [], [], []
    for i in range(max_eval + 1):
        if is_pivot_high(h, i, pivot_len, n):
            events_idx.append(i + pivot_len); events_val.append(h[i]); events_is_high.append(True)
        if is_pivot_low(l, i, pivot_len, n):
            events_idx.append(i + pivot_len); events_val.append(l[i]); events_is_high.append(False)

    order = sorted(range(len(events_idx)), key=lambda k: events_idx[k])
    cur_high = np.nan
    cur_low = np.nan
    cursor = 0
    for idx in range(n):
        while cursor < len(order) and events_idx[order[cursor]] == idx:
            k = order[cursor]
            if events_is_high[k]:
                cur_high = events_val[k]
            else:
                cur_low = events_val[k]
            cursor += 1
        last_swing_high[idx] = cur_high
        last_swing_low[idx] = cur_low
    return last_swing_high, last_swing_low


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

def compute_trend_bos_choch(h, l, c, pivot_len):
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

    for i in range(max_eval+1):
        ph = is_pivot_high(h, i, pivot_len, n)
        pl = is_pivot_low(l, i, pivot_len, n)
        if ph:
            piv_at.append(i+pivot_len); piv_val.append(h[i]); piv_is_high.append(True)
        if pl:
            piv_at.append(i+pivot_len); piv_val.append(l[i]); piv_is_high.append(False)
        while piv_cursor < len(piv_at) and piv_at[piv_cursor] == i:
            if piv_is_high[piv_cursor]: sh = piv_val[piv_cursor]; have_sh = True
            else: sl = piv_val[piv_cursor]; have_sl = True
            piv_cursor += 1
        if cur <= 0 and have_sh and c[i] > sh: cur = 1
        elif cur >= 0 and have_sl and c[i] < sl: cur = -1
        trend[i] = cur

    return trend

def compute_trend_choch_then_bos(h, l, c, pivot_len, hold_last):
    n = len(c)
    trend = np.full(n, np.nan, dtype=float)
    if n < pivot_len*2+5:
        return trend

    sh = sl = 0.0
    have_sh = have_sl = False
    confirmed = 0
    pending_dir = 0
    choch_ref = 0.0

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
            if piv_is_high[piv_cursor]: sh = piv_val[piv_cursor]; have_sh = True
            else: sl = piv_val[piv_cursor]; have_sl = True
            piv_cursor += 1

        close_i = c[i]

        if pending_dir == 0:
            if confirmed <= 0 and have_sh and close_i > sh:
                pending_dir = 1
                choch_ref = sh
            elif confirmed >= 0 and have_sl and close_i < sl:
                pending_dir = -1
                choch_ref = sl
        elif pending_dir == 1:
            if have_sl and close_i < sl:
                pending_dir = 0
            elif have_sh and close_i > sh and sh != choch_ref:
                confirmed = 1
                pending_dir = 0
        elif pending_dir == -1:
            if have_sh and close_i > sh:
                pending_dir = 0
            elif have_sl and close_i < sl and sl != choch_ref:
                confirmed = -1
                pending_dir = 0

        if pending_dir != 0 and not hold_last:
            painted_trend = 0
        else:
            painted_trend = confirmed
        trend[i] = painted_trend

    return trend

def compute_trend(df, engine, pivot_len, req_pairs=2, hold_last=False):
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    if engine == "PIVOT_PAIRS":
        return compute_trend_pivot_pairs(o, h, l, c, pivot_len, req_pairs, hold_last)
    elif engine == "BOS_CHOCH":
        return compute_trend_bos_choch(h, l, c, pivot_len)
    elif engine == "CHOCH_THEN_BOS":
        return compute_trend_choch_then_bos(h, l, c, pivot_len, hold_last)
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

    return eq, prem_top, prem_bot, disc_top, disc_bot, valid


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
# ---------------------------------------------------------------------------

def compute_sma_trend_filter(df, fast_period, slow_period, signal_period,
                             range_period, mult):
    close = df["close"]
    fast_sma = close.rolling(fast_period).mean()
    slow_sma = close.rolling(slow_period).mean()

    osc = ((fast_sma - slow_sma) / slow_sma) * 100.0

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

    return osc, signal, range_level, color


# ---------------------------------------------------------------------------
# Pattern detection – MQL5-exact Japanese patterns only
# ---------------------------------------------------------------------------

def _body_ratio_mask(o, h, l, c, ratio):
    r = h - l
    body = np.abs(c - o)
    return np.where(r > 0, body / r >= ratio, False)

def _body_ratio_most_mask(o, h, l, c, ratio):
    r = h - l
    body = np.abs(c - o)
    return np.where(r > 0, body / r <= ratio, True)

def detect_mql5_patterns(df, qualified_mask, htf_trend, require_htf_agree):
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    n = len(df)

    bear_now   = c < o
    bull_now   = c > o
    doji_shape = _body_ratio_most_mask(o, h, l, c, config.PATTERN_DOJI_BODY_RATIO)

    body_ratio = np.where(h - l > 0, np.abs(c - o) / (h - l), np.nan)
    doji_ratio = config.PATTERN_DOJI_BODY_RATIO
    spin_ratio = config.PATTERN_SPINNING_TOP_BODY_RATIO
    spin_shape = (body_ratio > doji_ratio) & (body_ratio <= spin_ratio)

    bear_prev = np.roll(bear_now, 1); bear_prev[0] = False
    bull_prev = np.roll(bull_now, 1); bull_prev[0] = False
    o_prev = np.roll(o, 1); c_prev = np.roll(c, 1); h_prev = np.roll(h, 1); l_prev = np.roll(l, 1)

    prev_body_ratio = np.where(h_prev - l_prev > 0, np.abs(c_prev - o_prev) / (h_prev - l_prev), np.nan)
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

    small_body = np.where(rng > 0, body / rng <= body_ratio_max, False)
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


def simulate_trade(hit, o, h, l, c, n, rr, use_breakeven, be_trigger, be_buffer,
                   overlapping, use_open_sl,
                   trail_mode="none", trail_start_after_breakeven=False,
                   trail_staircase_steps=None,
                   trail_atr=None, trail_atr_mult=2.0, trail_atr_min_r=0.0,
                   trail_swing_high=None, trail_swing_low=None,
                   trail_structure_buffer_pct=0.0):
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
            "max_favorable_r_b4_sl": None,
            "entry": entry,
            "sl": sl,
            "tp": None,
            "exit_price": None,
            "exit_time": None,
            "breakeven_hit": False,
            "trail_active": False,
        }

    tp = entry + rr * risk if is_bull else entry - rr * risk

    be_stop = entry + be_buffer * risk if is_bull else entry - be_buffer * risk
    be_trigger_price = entry + be_trigger * risk if is_bull else entry - be_trigger * risk

    current_sl = sl
    breakeven_armed = False
    trail_active = False
    max_favorable_r = 0.0

    for idx in range(entry_idx + 1, n):
        bar_high = float(h[idx])
        bar_low = float(l[idx])

        fav_extreme = bar_high if is_bull else bar_low
        bar_fav_r = (fav_extreme - entry) / risk if is_bull else (entry - fav_extreme) / risk
        if bar_fav_r > max_favorable_r:
            max_favorable_r = bar_fav_r

        if use_breakeven and not breakeven_armed:
            if is_bull and bar_high >= be_trigger_price:
                breakeven_armed = True
            elif not is_bull and bar_low <= be_trigger_price:
                breakeven_armed = True

        if breakeven_armed:
            hit_be = (bar_low <= be_stop) if is_bull else (bar_high >= be_stop)
            if hit_be:
                exit_price = be_stop
                r = (exit_price - entry) / risk if is_bull else (entry - exit_price) / risk
                return {
                    "outcome": "breakeven",
                    "r_multiple": round(r, 4),
                    "max_favorable_r": round(max_favorable_r, 4),
                    "max_favorable_r_b4_sl": None,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "exit_price": exit_price,
                    "exit_index": idx,
                    "breakeven_hit": True,
                    "trail_active": trail_active,
                }

        # --- Trailing stop update: SL only ever tightens, never loosens ---
        if trail_mode != "none":
            trail_ok = breakeven_armed if trail_start_after_breakeven else True
            if trail_ok:
                new_sl = None
                if trail_mode == "staircase" and trail_staircase_steps:
                    for trigger_r, lock_r in trail_staircase_steps:
                        if max_favorable_r >= trigger_r:
                            cand = entry + lock_r * risk if is_bull else entry - lock_r * risk
                            if new_sl is None or (is_bull and cand > new_sl) or (not is_bull and cand < new_sl):
                                new_sl = cand
                elif trail_mode == "atr" and trail_atr is not None:
                    if max_favorable_r >= trail_atr_min_r:
                        atr_val = trail_atr[idx]
                        if not np.isnan(atr_val):
                            new_sl = (fav_extreme - atr_val * trail_atr_mult) if is_bull \
                                      else (fav_extreme + atr_val * trail_atr_mult)
                elif trail_mode == "structure" and trail_swing_low is not None and trail_swing_high is not None:
                    if is_bull:
                        sw = trail_swing_low[idx]
                        if not np.isnan(sw):
                            new_sl = sw - entry * (trail_structure_buffer_pct / 100.0)
                    else:
                        sw = trail_swing_high[idx]
                        if not np.isnan(sw):
                            new_sl = sw + entry * (trail_structure_buffer_pct / 100.0)

                if new_sl is not None:
                    if is_bull and new_sl > current_sl:
                        current_sl = new_sl
                        trail_active = True
                    elif not is_bull and new_sl < current_sl:
                        current_sl = new_sl
                        trail_active = True

        if use_open_sl:
            sl_hit = (c[idx] <= current_sl) if is_bull else (c[idx] >= current_sl)
        else:
            sl_hit = (bar_low <= current_sl) if is_bull else (bar_high >= current_sl)

        if sl_hit:
            exit_price = c[idx] if use_open_sl else current_sl
            r = (exit_price - entry) / risk if is_bull else (entry - exit_price) / risk
            return {
                "outcome": "trail" if trail_active else "sl",
                "r_multiple": round(r, 4),
                "max_favorable_r": round(max_favorable_r, 4),
                "max_favorable_r_b4_sl": None,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit_price": exit_price,
                "exit_index": idx,
                "breakeven_hit": False,
                "trail_active": trail_active,
            }

        tp_hit = (bar_high >= tp) if is_bull else (bar_low <= tp)
        if tp_hit:
            return {
                "outcome": "tp",
                "r_multiple": round(rr, 4),
                "max_favorable_r": round(max_favorable_r, 4),
                "max_favorable_r_b4_sl": None,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit_price": tp,
                "exit_index": idx,
                "breakeven_hit": False,
                "trail_active": trail_active,
            }

    return {
        "outcome": "open",
        "r_multiple": None,
        "max_favorable_r": round(max_favorable_r, 4),
        "max_favorable_r_b4_sl": None,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "exit_price": None,
        "exit_index": None,
        "breakeven_hit": breakeven_armed,
        "trail_active": trail_active,
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
BG_HIGHLIGHT = "\033[1;37;44m"  # bold white text on blue background
MAX_FAV_R_HIGHLIGHT_THRESHOLD = 9.0

def box_line(line: str, width: int = 100):
    print(f"│ {line[:width-2]:<{width-2}} │")

def print_detailed_trades(hits, ctf_df, trail_atr=None, trail_swing_high=None, trail_swing_low=None):
    n = len(ctf_df)
    o = ctf_df["open"].to_numpy()
    h = ctf_df["high"].to_numpy()
    l = ctf_df["low"].to_numpy()
    c = ctf_df["close"].to_numpy()

    overlapping = config.OVERLAPPING_TRADES
    blocked_until_idx = -1

    trail_mode = getattr(config, "TRAIL_MODE", "none")

    for hit in hits:
        if not overlapping and hit["bar_index"] < blocked_until_idx:
            hit["trade"] = {
                "outcome": "skipped_overlap", "r_multiple": None,
                "max_favorable_r": None, "max_favorable_r_b4_sl": None,
                "entry": None, "sl": None, "tp": None,
                "exit_price": None, "exit_time": None,
                "breakeven_hit": False, "trail_active": False,
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
            trail_mode=trail_mode,
            trail_start_after_breakeven=getattr(config, "TRAIL_START_AFTER_BREAKEVEN", False),
            trail_staircase_steps=getattr(config, "TRAIL_STAIRCASE_STEPS", None),
            trail_atr=trail_atr,
            trail_atr_mult=getattr(config, "TRAIL_ATR_MULTIPLIER", 2.0),
            trail_atr_min_r=getattr(config, "TRAIL_ATR_MIN_ACTIVATE_R", 0.0),
            trail_swing_high=trail_swing_high,
            trail_swing_low=trail_swing_low,
            trail_structure_buffer_pct=getattr(config, "TRAIL_STRUCTURE_BUFFER_PCT", 0.0),
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
        if config.USE_COMMISSION and trade.get("outcome") in ("tp","sl","breakeven","trail"):
            trade["r_multiple_gross"] = trade["r_multiple"]
            trade["r_multiple"] = round(trade["r_multiple"] - config.COMMISSION_R, 4)
        hit["trade"] = trade

    W_SN = 5; W_TIME = 19; W_PATTERN = 15; W_DIR = 5; W_ENTRY = 13
    W_OUTCOME = 5; W_OUTCOME_R = 6; W_MFE = 8; W_MFEb4 = 12; W_CUM = 10

    show_outcome_r = getattr(config, "SHOW_OUTCOME_R", True)

    if show_outcome_r:
        header_fields = [
            f"{'S/N':<{W_SN}}", f"{'DateTime':<{W_TIME}}", f"{'Pattern':<{W_PATTERN}}",
            f"{'Dir':<{W_DIR}}", f"{'Entry':>{W_ENTRY}}", f"{'Out':<{W_OUTCOME}}",
            f"{'Out R':>{W_OUTCOME_R}}", f"{'MaxFavR':>{W_MFE}}", f"{'MaxFavR_b4SL':>{W_MFEb4}}",
            f"{'Cum R':>{W_CUM}}",
        ]
    else:
        header_fields = [
            f"{'S/N':<{W_SN}}", f"{'DateTime':<{W_TIME}}", f"{'Pattern':<{W_PATTERN}}",
            f"{'Dir':<{W_DIR}}", f"{'Entry':>{W_ENTRY}}", f"{'Out':<{W_OUTCOME}}",
            f"{'MaxFavR':>{W_MFE}}", f"{'MaxFavR_b4SL':>{W_MFEb4}}", f"{'Cum R':>{W_CUM}}",
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
        elif outcome == "trail": display_outcome = "TRAIL"
        elif outcome == "skipped_overlap": display_outcome = "SKIP"

        if outcome in ("tp", "breakeven"): out_color = GREEN
        elif outcome == "trail": out_color = (GREEN if (r_val or 0) > 0 else CYAN)
        elif outcome == "sl": out_color = RED
        elif outcome == "skipped_overlap": out_color = YELLOW
        else: out_color = RESET
        out_colored = f"{out_color}{display_outcome[:W_OUTCOME]:<{W_OUTCOME}}{RESET}"

        if r_val is not None:
            out_r_color = GREEN if r_val > 0 else (RED if r_val < 0 else RESET)
            out_r_str = f"{r_val:+.2f}R"
        else:
            out_r_color = RESET; out_r_str = "  --"
        out_r_colored = f"{out_r_color}{out_r_str:>{W_OUTCOME_R}}{RESET}"

        cum_color = GREEN if net_r > 0 else (RED if net_r < 0 else RESET)
        cum_colored = f"{cum_color}{cum_str:>{W_CUM}}{RESET}"

        dir_str = f"{hit['direction']:<{W_DIR}}"

        if show_outcome_r:
            row = (f"│ {sn:<{W_SN}} {hit['time']:<{W_TIME}} {pat_colored} {dir_str} "
                   f"{entry_str} {out_colored} {out_r_colored} {mfe_str} "
                   f"{mfe_b4_str} {cum_colored} │")
        else:
            row = (f"│ {sn:<{W_SN}} {hit['time']:<{W_TIME}} {pat_colored} {dir_str} "
                   f"{entry_str} {out_colored} {mfe_str} "
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


def print_performance_stats(hits, net_r):
    if not hits:
        print("\n[Performance] No trades found. Cannot compute performance statistics.")
        return {
            "n_closed": 0, "n_open": 0, "n_wins": 0, "n_losses": 0,
            "win_rate": 0.0, "total_r": 0.0, "expectancy": 0.0,
            "n_tp": 0, "n_sl": 0, "n_be": 0, "n_trail": 0, "max_dd": 0.0,
            "worst_month_dd": 0.0,
        }

    closed = [h["trade"] for h in hits if h["trade"]["outcome"] not in ("open","invalid","skipped_overlap")]
    outcomes = [t["outcome"] for t in closed]
    r_vals = [t["r_multiple"] for t in closed if t["r_multiple"] is not None]

    n_tp = outcomes.count("tp")
    n_sl = outcomes.count("sl")
    n_be = outcomes.count("breakeven")
    n_trail = outcomes.count("trail")
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
    print(f"  Outcomes -> TP: {n_tp}   SL: {n_sl}   Breakeven: {n_be}   Trail: {n_trail}")
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
        months.setdefault(m, {"trades": 0, "wins": 0, "losses": 0, "r": 0.0, "peak": 0.0, "dd": 0.0})
        months[m]["trades"] += 1
        if t.get("r_multiple", 0) > 0:
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
    box_line("Month       Trades   Wins   Losses   WinRate     Month R      Max DD       Cum R")
    print("│ " + "─"*97 + "│")
    cum = 0.0
    for m in sorted(months):
        mo = months[m]
        wr = mo["wins"] / mo["trades"] * 100 if mo["trades"] else 0
        cum += mo["r"]
        box_line(f"{m:<7} {mo['trades']:>7} {mo['wins']:>6} {mo['losses']:>8} {wr:>7.2f}% {mo['r']:>11.2f}R {mo['dd']:>11.2f}R {cum:>11.2f}R")
    print("└" + "─"*100 + "┘")

    print("\n" + "-"*70)
    print("Per‑pattern trade statistics:")
    print("-"*70)
    pattern_stats = {}
    for hit in hits:
        pat = hit["pattern"]
        t = hit["trade"]
        if pat not in pattern_stats:
            pattern_stats[pat] = {"tp":0,"sl":0,"be":0,"trail":0,"open":0,"r_sum":0.0,"wins":0,"count":0}
        ps = pattern_stats[pat]
        if t["outcome"] in ("tp","sl","breakeven","trail"):
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
        print(f"  TP: {ps['tp']}   SL: {ps['sl']}   BE: {ps['be']}   Trail: {ps['trail']}   Open: {ps['open']}")
        if ps["count"]:
            print(f"  Win rate: {wr:.2f}%   Total R: {ps['r_sum']:+.2f}   Expectancy: {ps['r_sum']/ps['count']:+.3f}R")
        else:
            print("  No closed trades")
    print("-"*70)

    worst_month_dd = max((mo["dd"] for mo in months.values()), default=0.0)

    return {
        "n_closed": n_closed, "n_open": n_open, "n_wins": n_wins, "n_losses": n_losses,
        "win_rate": win_rate, "total_r": total_r, "expectancy": expectancy,
        "n_tp": n_tp, "n_sl": n_sl, "n_be": n_be, "n_trail": n_trail, "max_dd": dd,
        "worst_month_dd": worst_month_dd,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scan(ctf_timeframe, is_multi=False):
    """Run the full scan + trade simulation for a single CTF timeframe.
    Mutates config.CTF_TIMEFRAME for the duration of the call (every helper
    below reads it from `config`), then returns a summary dict for cross-
    timeframe comparison."""
    config.CTF_TIMEFRAME = ctf_timeframe

    print(f"[Scanner] symbol={config.SYMBOL}  ctf_tf={config.CTF_TIMEFRAME}")
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
                              config.REQUIRED_PIVOT_PAIRS, config.HOLD_LAST_TREND_ON_UNDEFINED)

    if config.INP_USE_HTF_ZONES:
        zone_htf_df = resample_to_tf(df, zone_htf_tf)
        if len(zone_htf_df) > 0:
            zone_delta = pd.Timedelta(zone_htf_df["time"].diff().mode().iloc[0])
            if zone_htf_df["time"].iloc[-1] + zone_delta > df["time"].iloc[-1]:
                zone_htf_df = zone_htf_df.iloc[:-1]
        print(f"[HTF Zones] {len(zone_htf_df)} bars ({zone_htf_tf})")

        htf_atr = precompute_atr(zone_htf_df, config.INP_ATR_PERIOD)
        eq_htf, pt_htf, pb_htf, dt_htf, db_htf, valid_htf = compute_pd_zones_vectorized(
            zone_htf_df, config.INP_LOOKBACK_PERIOD, config.INP_USE_ATR,
            config.INP_ATR_PERIOD, config.INP_ATR_MULTIPLIER,
            config.INP_BUFFER_POINTS, config.INP_USE_MANUAL_ZONES,
            config.INP_DISCOUNT_INNER_OFFSET, config.INP_DISCOUNT_OUTER_OFFSET,
            config.INP_PREMIUM_INNER_OFFSET, config.INP_PREMIUM_OUTER_OFFSET,
            point, htf_atr)
        eq, pt, pb, dt, db, zone_valid = map_htf_zones_to_ctf(zone_htf_df, df,
                                                              (eq_htf, pt_htf, pb_htf,
                                                               dt_htf, db_htf, valid_htf))
    else:
        print("[Indicator] computing zones on CTF")
        ctf_atr = precompute_atr(df, config.INP_ATR_PERIOD)
        eq, pt, pb, dt, db, zone_valid = compute_pd_zones_vectorized(
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
                                  config.REQUIRED_PIVOT_PAIRS, config.HOLD_LAST_TREND_ON_UNDEFINED)
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
            config.SMA_RANGE_FILTER_MULTIPLIER
        )
        print(f"[SMA Filter] enabled, colour distribution: "
              f"bull={int((sma_color == 0).sum())}, bear={int((sma_color == 1).sum())}, "
              f"range={int((sma_color == 2).sum())}")
    else:
        sma_color = np.full(len(df), 2, dtype=int)

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

    trail_mode = getattr(config, "TRAIL_MODE", "none")
    trail_atr = None
    trail_swing_high = None
    trail_swing_low = None
    if trail_mode == "atr":
        trail_atr = precompute_atr(df, getattr(config, "TRAIL_ATR_PERIOD", 14))
    elif trail_mode == "structure":
        trail_swing_high, trail_swing_low = compute_swing_levels(
            df["high"].values, df["low"].values, config.PIVOT_LENGTH)

    sim_desc = [
        f"RR={config.RR}",
        f"breakeven={'ON' if config.USE_BREAKEVEN else 'OFF'}",
        f"overlapping={'ON' if config.OVERLAPPING_TRADES else 'OFF'}",
        f"trail={trail_mode}",
    ]
    if trail_mode != "none" and getattr(config, "TRAIL_START_AFTER_BREAKEVEN", False):
        sim_desc.append("trail-after-BE")
    if config.USE_OPEN_SL:
        sim_desc.append("open-SL")
    print(f"[Scanner] simulating trades: {', '.join(sim_desc)}")

    net_r = print_detailed_trades(all_hits, df, trail_atr=trail_atr,
                                   trail_swing_high=trail_swing_high,
                                   trail_swing_low=trail_swing_low)
    stats = print_performance_stats(all_hits, net_r)

    out_name = f"{config.SYMBOL}_{ctf_timeframe}.json" if is_multi else \
        (config.RESULT_FILENAME if config.RESULT_FILENAME else f"{config.SYMBOL}.json")
    out_path = os.path.join(config.RESULT_DIR, out_name)
    output = {
        "symbol": config.SYMBOL,
        "ctf_timeframe": config.CTF_TIMEFRAME,
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


def main():
    timeframes = getattr(config, "TIMEFRAMES", None)
    if not timeframes:
        timeframes = [config.CTF_TIMEFRAME]
    is_multi = len(timeframes) > 1

    original_ctf = config.CTF_TIMEFRAME

    if os.path.exists(config.RESULT_DIR):
        shutil.rmtree(config.RESULT_DIR)
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    summaries = []
    for tf in timeframes:
        if is_multi:
            print("\n" + "#"*100)
            print(f"#  TIMEFRAME: {tf}")
            print("#"*100)
        summary = run_scan(tf, is_multi=is_multi)
        summaries.append(summary)

    config.CTF_TIMEFRAME = original_ctf

    if is_multi:
        print_timeframe_comparison(summaries)


if __name__ == "__main__":
    main()