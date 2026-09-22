"""
main.py -- HTF Pattern Scanner (optimized)
Filters exactly as TrendStructure_PDZones_Combined, then pattern detection
and trade simulation.  Speed improvements: precomputed ATR, vectorized zones,
searchsorted mapping, vectorized pattern detection, merged MFE/SL scan.
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
        "1min": "5min", "5min": "15min", "15min": "30min", "30min": "1h",
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
# Pre‑compute ATR (SMA of true range) – called once per dataset
# ---------------------------------------------------------------------------

def precompute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    """Return array of same length as df, NaN where i < period-1."""
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
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1:] = pd.Series(tr).rolling(period).mean().values[period-1:]
    return atr


# ---------------------------------------------------------------------------
# Trend engines (unchanged logic, using numpy arrays directly)
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
    trend = np.zeros(n, dtype=int)
    if n < pivot_len*2+5: return trend
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
    for idx in range(n):
        while apply_cursor < len(pending_at) and pending_at[apply_cursor] == idx:
            applied = pending_val[apply_cursor]; apply_cursor += 1
        trend[idx] = applied
    return trend

def compute_trend_bos_choch(h, l, c, pivot_len):
    n = len(c)
    trend = np.zeros(n, dtype=int)
    if n < pivot_len*2+5: return trend
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
    for i in range(max_eval+1, n): trend[i] = cur if (have_sh or have_sl) else 0
    return trend

def compute_trend_choch_then_bos(h, l, c, pivot_len):
    n = len(c)
    trend = np.zeros(n, dtype=int)
    if n < pivot_len*2+5: return trend
    sh = sl = 0.0
    have_sh = have_sl = False
    confirmed = 0; pending_dir = 0; choch_ref = 0.0
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
                pending_dir=1; choch_ref=sh
            elif confirmed >= 0 and have_sl and close_i < sl:
                pending_dir=-1; choch_ref=sl
        elif pending_dir == 1:
            if have_sl and close_i < sl: pending_dir=0
            elif have_sh and close_i > sh and sh != choch_ref:
                confirmed=1; pending_dir=0
        elif pending_dir == -1:
            if have_sh and close_i > sh: pending_dir=0
            elif have_sl and close_i < sl and sl != choch_ref:
                confirmed=-1; pending_dir=0
        trend[i] = confirmed if pending_dir==0 else 0
    for i in range(max_eval+1, n): trend[i] = confirmed if pending_dir==0 else 0
    return trend

def compute_trend(df, engine, pivot_len, req_pairs=2, hold_last=False):
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    if engine == "PIVOT_PAIRS":
        return compute_trend_pivot_pairs(o, h, l, c, pivot_len, req_pairs, hold_last)
    elif engine == "BOS_CHOCH":
        return compute_trend_bos_choch(h, l, c, pivot_len)
    elif engine == "CHOCH_THEN_BOS":
        return compute_trend_choch_then_bos(h, l, c, pivot_len)
    else:
        raise ValueError(f"Unknown engine: {engine}")


# ---------------------------------------------------------------------------
# PD Zones – fully vectorized
# ---------------------------------------------------------------------------

def compute_pd_zones_vectorized(df, lookback, use_atr, atr_period, atr_mult,
                                buffer_points, use_manual,
                                disc_inner, disc_outer, prem_inner, prem_outer,
                                point_size, atr_array=None):
    """Returns eq, prem_top, prem_bot, disc_top, disc_bot as numpy arrays."""
    h = df["high"].values; l = df["low"].values
    n = len(df)
    roll_max = pd.Series(h).rolling(lookback, min_periods=lookback).max().values
    roll_min = pd.Series(l).rolling(lookback, min_periods=lookback).min().values
    mid = (roll_max + roll_min) / 2.0

    if use_manual:
        dt = mid - disc_inner
        db = mid - disc_outer
        pb = mid + prem_inner
        pt = mid + prem_outer
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

    eq = np.where(np.isnan(roll_max), np.nan, mid)
    prem_top = np.where(np.isnan(roll_max), np.nan, pt)
    prem_bot = np.where(np.isnan(roll_max), np.nan, pb)
    disc_top = np.where(np.isnan(roll_max), np.nan, dt)
    disc_bot = np.where(np.isnan(roll_max), np.nan, db)

    return eq, prem_top, prem_bot, disc_top, disc_bot


# ---------------------------------------------------------------------------
# HTF‑to‑CTF mapping (searchsorted)
# ---------------------------------------------------------------------------

def map_htf_zones_to_ctf(htf_df, ctf_df, htf_zone_arrays):
    eq, pt, pb, dt, db = htf_zone_arrays
    htf_times = htf_df["time"].values
    ctf_times = ctf_df["time"].values
    idx = np.searchsorted(htf_times, ctf_times, side='right') - 1
    idx = np.clip(idx, 0, len(eq)-1)
    return eq[idx], pt[idx], pb[idx], dt[idx], db[idx]


# ---------------------------------------------------------------------------
# Pattern detection – vectorized masks + priority suppression
# ---------------------------------------------------------------------------

def _body_ratio_mask(o, h, l, c, ratio):
    r = h - l
    body = np.abs(c - o)
    return np.where(r > 0, body / r >= ratio, False)

def _body_ratio_most_mask(o, h, l, c, ratio):
    r = h - l
    body = np.abs(c - o)
    return np.where(r > 0, body / r <= ratio, True)

def detect_japanese_hits(df, qualified_mask, htf_trend, require_htf_agree):
    """Return list of hit dicts for Japanese patterns, exactly as MQL5 priority."""
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    n = len(df)

    bear_now   = c < o
    bull_now   = c > o
    doji_shape = _body_ratio_most_mask(o, h, l, c, config.PATTERN_DOJI_BODY_RATIO)
    spin_shape = _body_ratio_mask(o, h, l, c, config.PATTERN_SPINNING_TOP_BODY_RATIO) & ~doji_shape

    bear_prev = np.roll(bear_now, 1); bear_prev[0] = False
    bull_prev = np.roll(bull_now, 1); bull_prev[0] = False
    o_prev = np.roll(o, 1); c_prev = np.roll(c, 1); h_prev = np.roll(h, 1); l_prev = np.roll(l, 1)

    pierce_mask = (
        bear_prev & bull_now &
        (l < l_prev) & (h <= h_prev)
    )
    dc_mask = (
        bull_prev & bear_now &
        (h > h_prev) & (l >= l_prev)
    )
    be_mask = (
        bear_prev & bull_now &
        ~doji_shape & ~spin_shape &
        (o < c_prev) & (c > o_prev) &
        (l < l_prev) & (h > h_prev)
    )
    bg_mask = (
        bull_prev & bear_now &
        ~doji_shape & ~spin_shape &
        (o > c_prev) & (c < o_prev) &
        (h > h_prev) & (l < l_prev)
    )
    cp_mask = (
        bear_prev & bull_now &
        ~doji_shape & ~spin_shape &
        (l < l_prev) & (c > c_prev)
    )
    cdc_mask = (
        bull_prev & bear_now &
        ~doji_shape & ~spin_shape &
        (h > h_prev) & (c < c_prev)
    )

    # Priority order: Piercing → DarkCloud → BullEngulf → BearEngulf → CustomPiercing → CustomDarkCloud
    pattern_code = np.zeros(n, dtype=int)
    pattern_code[pierce_mask] = 1
    dc_not_pierce = dc_mask & (pattern_code == 0)
    pattern_code[dc_not_pierce] = 2
    be_not_prev = be_mask & (pattern_code == 0)
    pattern_code[be_not_prev] = 3
    bg_not_prev = bg_mask & (pattern_code == 0)
    pattern_code[bg_not_prev] = 4
    cp_not_prev = cp_mask & (pattern_code == 0)
    pattern_code[cp_not_prev] = 5
    cdc_not_prev = cdc_mask & (pattern_code == 0)
    pattern_code[cdc_not_prev] = 6

    code_info = {
        1: ("Piercing", True, 2),
        2: ("DarkCloud", False, 2),
        3: ("BullEngulf", True, 2),
        4: ("BearEngulf", False, 2),
        5: ("CustomPiercing", True, 2),
        6: ("CustomDarkCloud", False, 2),
    }

    hits = []
    times = df["time"]
    indices = np.where((pattern_code > 0) & qualified_mask)[0]
    for i in indices:
        code = pattern_code[i]
        name, is_bull, span = code_info[code]
        if require_htf_agree:
            lo = max(0, i - span + 1)
            wanted_trend = 1 if is_bull else -1
            align = np.any(htf_trend[lo:i+1] == wanted_trend)
            if not align:
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
    return hits


def detect_doji_spinning_top_hits(df, qualified_mask, htf_trend, require_htf_agree):
    """Pending‑confirm Doji/SpinningTop."""
    o = df["open"].values; h = df["high"].values; l = df["low"].values; c = df["close"].values
    n = len(df)
    doji_shape = _body_ratio_most_mask(o, h, l, c, config.PATTERN_DOJI_BODY_RATIO)
    spin_shape = _body_ratio_mask(o, h, l, c, config.PATTERN_SPINNING_TOP_BODY_RATIO) & ~doji_shape

    hits = []
    pending = False
    pending_is_doji = False
    pending_high = 0.0; pending_low = 0.0

    for i in range(n):
        if pending:
            confirm_close = c[i]
            if confirm_close > pending_high or confirm_close < pending_low:
                if qualified_mask[i]:
                    is_bull = confirm_close > pending_high
                    align = (not require_htf_agree) or (
                        (is_bull and htf_trend[i] == 1) or
                        ((not is_bull) and htf_trend[i] == -1)
                    )
                    if align:
                        name = "Doji" if pending_is_doji else "SpinningTop"
                        sl_extreme = pending_low if is_bull else pending_high
                        hits.append({
                            "time": df["time"].iloc[i].strftime("%Y-%m-%d %H:%M:%S"),
                            "bar_index": i,
                            "pattern": name,
                            "direction": "bull" if is_bull else "bear",
                            "open": float(o[i]), "high": float(h[i]),
                            "low": float(l[i]), "close": float(confirm_close),
                            "sl_extreme": sl_extreme,
                            "library": "japanese",
                        })
            pending = False

        if qualified_mask[i]:
            if doji_shape[i]:
                pending = True; pending_is_doji = True
                pending_high = float(h[i]); pending_low = float(l[i])
            elif spin_shape[i]:
                pending = True; pending_is_doji = False
                pending_high = float(h[i]); pending_low = float(l[i])
    return hits


def detect_candle_count_hits(df, qualified_mask, htf_trend, require_htf_agree):
    """Candle-count patterns (2Bull, 2Bear, 3Bull, 3Bear) – vectorized."""
    o = df["open"].values; h = df["high"].values; l = df["low"].values; c = df["close"].values
    n = len(df)
    bear_now = c < o; bull_now = c > o
    bear_prev = np.roll(bear_now, 1); bear_prev[0] = False
    bull_prev = np.roll(bull_now, 1); bull_prev[0] = False
    h_prev = np.roll(h, 1); l_prev = np.roll(l, 1)

    two_bull = bear_prev & bull_now & (c > h_prev)
    two_bear = bull_prev & bear_now & (c < l_prev)

    bear2  = np.roll(bear_now, 2); bear2[0] = bear2[1] = False
    bull1  = np.roll(bull_now, 1); bull1[0] = False
    c1 = np.roll(c, 1); h0 = np.roll(h, 2)
    three_bull = bear2 & bull1 & bull_now & (c1 <= h0) & (c > h_prev)

    bull2  = np.roll(bull_now, 2); bull2[0] = bull2[1] = False
    bear1  = np.roll(bear_now, 1); bear1[0] = False
    l0 = np.roll(l, 2)
    three_bear = bull2 & bear1 & bear_now & (c1 >= l0) & (c < l_prev)

    hits = []
    times = df["time"]
    for idx, name, is_bull, span in [
        (two_bull, "2Bull", True, 2),
        (two_bear, "2Bear", False, 2),
        (three_bull, "3Bull", True, 3),
        (three_bear, "3Bear", False, 3),
    ]:
        mask = idx & qualified_mask
        indices = np.where(mask)[0]
        for i in indices:
            if require_htf_agree:
                lo = max(0, i - span + 1)
                wanted_trend = 1 if is_bull else -1
                align = np.any(htf_trend[lo:i+1] == wanted_trend)
                if not align: continue
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
                "library": "candle_count",
            })
    return hits


# ---------------------------------------------------------------------------
# Trade simulation – uses precomputed ATR, merged MFE/SL scan
# ---------------------------------------------------------------------------

def simulate_trade(hit, o, h, l, c, n, rr, use_breakeven, be_trigger, be_buffer,
                   use_atr_trail, atr_trigger, atr_period, atr_mult,
                   overlapping, use_open_sl, atr_array):
    entry_idx = hit["bar_index"]
    entry = float(c[entry_idx])
    is_bull = hit["direction"] == "bull"
    sl = hit["sl_extreme"]
    risk = abs(entry - sl)
    if risk <= 0:
        return {"outcome": "invalid", "r_multiple": 0.0, "max_favorable_r": 0.0,
                "entry": entry, "sl": sl, "tp": None, "exit_price": None,
                "exit_time": None, "breakeven_hit": False, "trail_active": False}

    if not use_atr_trail:
        tp = entry + rr * risk if is_bull else entry - rr * risk
    else:
        tp = None

    be_stop = entry + be_buffer * risk if is_bull else entry - be_buffer * risk
    be_trigger_price = entry + be_trigger * risk if is_bull else entry - be_trigger * risk

    current_sl = sl
    breakeven_armed = False
    trail_armed = False
    max_favorable_r = 0.0

    for idx in range(entry_idx + 1, n):
        bar_high = float(h[idx]); bar_low = float(l[idx])

        # MFE
        fav_extreme = bar_high if is_bull else bar_low
        bar_fav_r = (fav_extreme - entry) / risk if is_bull else (entry - fav_extreme) / risk
        if bar_fav_r > max_favorable_r:
            max_favorable_r = bar_fav_r

        # ATR trailing
        if use_atr_trail and not trail_armed and max_favorable_r >= atr_trigger:
            trail_armed = True
            atr_val = atr_array[idx]
            if np.isnan(atr_val): continue
            if is_bull:
                trail_stop = bar_high - atr_mult * atr_val
            else:
                trail_stop = bar_low + atr_mult * atr_val
            if (is_bull and trail_stop > current_sl) or (not is_bull and trail_stop < current_sl):
                current_sl = trail_stop

        if trail_armed:
            atr_val = atr_array[idx]
            if not np.isnan(atr_val):
                if is_bull:
                    candidate = bar_high - atr_mult * atr_val
                    if candidate > current_sl: current_sl = candidate
                else:
                    candidate = bar_low + atr_mult * atr_val
                    if candidate < current_sl: current_sl = candidate

        # Breakeven arming
        if not trail_armed and use_breakeven and not breakeven_armed:
            if is_bull and bar_high >= be_trigger_price:
                breakeven_armed = True
            elif not is_bull and bar_low <= be_trigger_price:
                breakeven_armed = True

        # Breakeven exit (hard)
        if breakeven_armed:
            hit_be = (bar_low <= be_stop) if is_bull else (bar_high >= be_stop)
            if hit_be:
                exit_price = be_stop
                r = (exit_price - entry) / risk if is_bull else (entry - exit_price) / risk
                return {
                    "outcome": "breakeven",
                    "r_multiple": round(r, 4),
                    "max_favorable_r": round(max_favorable_r, 4),
                    "entry": entry, "sl": sl, "tp": tp,
                    "exit_price": exit_price,
                    "exit_index": idx,
                    "breakeven_hit": True,
                    "trail_active": trail_armed,
                }

        # Normal SL
        if use_open_sl:
            sl_hit = (c[idx] <= current_sl) if is_bull else (c[idx] >= current_sl)
        else:
            sl_hit = (bar_low <= current_sl) if is_bull else (bar_high >= current_sl)

        if sl_hit:
            exit_price = c[idx] if use_open_sl else current_sl
            r = (exit_price - entry) / risk if is_bull else (entry - exit_price) / risk
            outcome = "trail" if trail_armed else "sl"
            return {
                "outcome": outcome,
                "r_multiple": round(r, 4),
                "max_favorable_r": round(max_favorable_r, 4),
                "entry": entry, "sl": sl, "tp": tp,
                "exit_price": exit_price,
                "exit_index": idx,
                "breakeven_hit": False,
                "trail_active": trail_armed,
            }

        # TP
        if not use_atr_trail and tp is not None:
            tp_hit = (bar_high >= tp) if is_bull else (bar_low <= tp)
            if tp_hit:
                return {
                    "outcome": "tp",
                    "r_multiple": round(rr, 4),
                    "max_favorable_r": round(max_favorable_r, 4),
                    "entry": entry, "sl": sl, "tp": tp,
                    "exit_price": tp,
                    "exit_index": idx,
                    "breakeven_hit": False,
                    "trail_active": trail_armed,
                }

    # End of data
    return {
        "outcome": "open",
        "r_multiple": None,
        "max_favorable_r": round(max_favorable_r, 4),
        "entry": entry, "sl": sl, "tp": tp,
        "exit_price": None, "exit_index": None,
        "breakeven_hit": breakeven_armed,
        "trail_active": trail_armed,
    }


# ---------------------------------------------------------------------------
# Reporting functions (unchanged from original, except simulate_trade call)
# ---------------------------------------------------------------------------

BOLD = "\033[1m"; GREEN = "\033[32m"; RED = "\033[31m"
YELLOW = "\033[33m"; CYAN = "\033[36m"; RESET = "\033[0m"

def box_line(line: str, width: int = 100):
    print(f"│ {line[:width-2]:<{width-2}} │")

def print_detailed_trades(hits, ctf_df, ctf_atr):
    n = len(ctf_df)
    o = ctf_df["open"].to_numpy()
    h = ctf_df["high"].to_numpy()
    l = ctf_df["low"].to_numpy()
    c = ctf_df["close"].to_numpy()

    overlapping = config.OVERLAPPING_TRADES
    blocked_until_idx = -1

    for hit in hits:
        if not overlapping and hit["bar_index"] < blocked_until_idx:
            hit["trade"] = {
                "outcome": "skipped_overlap", "r_multiple": None,
                "max_favorable_r": None, "entry": None, "sl": None, "tp": None,
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
            use_atr_trail=config.USE_ATR_TRAIL,
            atr_trigger=config.ATR_TRAIL_TRIGGER,
            atr_period=config.ATR_TRAIL_PERIOD,
            atr_mult=config.ATR_TRAIL_MULT,
            overlapping=overlapping,
            use_open_sl=config.USE_OPEN_SL,
            atr_array=ctf_atr,
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
        mfe_str = f"{mfe:+.2f}R" if mfe is not None else "  --"
        if r_val is not None:
            net_r += r_val
        cum_str = f"{net_r:+.2f}R"

        pat_color = GREEN if hit["direction"] == "bull" else RED
        pat_colored = f"{pat_color}{hit['pattern'][:W_PATTERN]:<{W_PATTERN}}{RESET}"

        display_outcome = outcome
        if outcome == "breakeven": display_outcome = "BE"
        elif outcome == "skipped_overlap": display_outcome = "SKIP"

        if outcome in ("tp", "breakeven"): out_color = GREEN
        elif outcome == "sl": out_color = RED
        elif outcome == "trail": out_color = GREEN if (r_val and r_val > 0) else RED
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
                   f"{entry_str} {out_colored} {out_r_colored} {mfe_str:>{W_MFE}} "
                   f"{'':>{W_MFEb4}} {cum_colored} │")
        else:
            row = (f"│ {sn:<{W_SN}} {hit['time']:<{W_TIME}} {pat_colored} {dir_str} "
                   f"{entry_str} {out_colored} {mfe_str:>{W_MFE}} "
                   f"{'':>{W_MFEb4}} {cum_colored} │")
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
    """Multi-row ASCII line chart of an equity curve, with a zero baseline
    and left-hand R-value axis labels. Returns a list of plain text rows
    (no box borders -- caller wraps each with box_line)."""
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

    trail_wins = sum(1 for t in closed if t["outcome"] == "trail" and t["r_multiple"] > 0)
    trail_losses = sum(1 for t in closed if t["outcome"] == "trail" and t["r_multiple"] <= 0)

    total_r = sum(r_vals)
    expectancy = total_r / len(r_vals) if r_vals else 0.0

    print("\n" + "="*70)
    print("Global trade statistics:")
    print("-"*70)
    skip_note = f"   (Skipped-overlap: {n_skipped})" if not config.OVERLAPPING_TRADES else ""
    print(f"  Total closed trades: {n_closed}   (Open: {n_open}){skip_note}")
    print(f"  Wins: {n_wins}   Losses: {n_losses}")
    if config.USE_ATR_TRAIL:
        print(f"  Trail exits: {n_trail}   (Trail wins: {trail_wins}   Trail losses: {trail_losses})")
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
            pattern_stats[pat] = {"tp":0,"sl":0,"be":0,"trail":0,"trail_w":0,"trail_l":0,"open":0,"r_sum":0.0,"wins":0,"count":0}
        ps = pattern_stats[pat]
        if t["outcome"] in ("tp","sl","breakeven","trail"):
            stat_key = "be" if t["outcome"] == "breakeven" else t["outcome"]
            ps[stat_key] = ps.get(stat_key, 0) + 1
            ps["count"] += 1
            if t["r_multiple"] is not None:
                ps["r_sum"] += t["r_multiple"]
                if t["r_multiple"] > 0: ps["wins"] += 1
                if t["outcome"] == "trail":
                    if t["r_multiple"] > 0: ps["trail_w"] += 1
                    else: ps["trail_l"] += 1
        elif t["outcome"] == "open":
            ps["open"] += 1
    for pat, ps in pattern_stats.items():
        wr = ps["wins"] / ps["count"] * 100 if ps["count"] else 0
        print(f"Pattern: {pat}")
        print(f"  TP: {ps['tp']}   SL: {ps['sl']}   BE: {ps['be']}   Trail: {ps['trail']} (W:{ps['trail_w']}/L:{ps['trail_l']})   Open: {ps['open']}")
        if ps["count"]:
            print(f"  Win rate: {wr:.2f}%   Total R: {ps['r_sum']:+.2f}   Expectancy: {ps['r_sum']/ps['count']:+.3f}R")
        else:
            print("  No closed trades")
    print("-"*70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
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

    if config.USE_HTF_MAPPING:
        htf_tf = map_timeframe(config.CTF_TIMEFRAME)
    else:
        htf_tf = "1h"
    print(f"[Indicator] Trend/Zones HTF: {htf_tf}")

    htf_df = resample_to_tf(df, htf_tf)
    print(f"[HTF] {len(htf_df)} HTF bars")

    print("[Indicator] computing HTF trend & zones (optimized)")
    htf_trend = compute_trend(htf_df, config.TREND_ENGINE, config.PIVOT_LENGTH,
                              config.REQUIRED_PIVOT_PAIRS, config.HOLD_LAST_TREND_ON_UNDEFINED)
    htf_atr = precompute_atr(htf_df, config.INP_ATR_PERIOD)

    if config.INP_USE_HTF_ZONES:
        eq_htf, pt_htf, pb_htf, dt_htf, db_htf = compute_pd_zones_vectorized(
            htf_df, config.INP_LOOKBACK_PERIOD, config.INP_USE_ATR,
            config.INP_ATR_PERIOD, config.INP_ATR_MULTIPLIER,
            config.INP_BUFFER_POINTS, config.INP_USE_MANUAL_ZONES,
            config.INP_DISCOUNT_INNER_OFFSET, config.INP_DISCOUNT_OUTER_OFFSET,
            config.INP_PREMIUM_INNER_OFFSET, config.INP_PREMIUM_OUTER_OFFSET,
            point, htf_atr)
        eq, pt, pb, dt, db = map_htf_zones_to_ctf(htf_df, df, (eq_htf, pt_htf, pb_htf, dt_htf, db_htf))
    else:
        print("[Indicator] computing CTF zones")
        ctf_atr = precompute_atr(df, config.INP_ATR_PERIOD)
        eq, pt, pb, dt, db = compute_pd_zones_vectorized(
            df, config.INP_LOOKBACK_PERIOD, config.INP_USE_ATR,
            config.INP_ATR_PERIOD, config.INP_ATR_MULTIPLIER,
            config.INP_BUFFER_POINTS, config.INP_USE_MANUAL_ZONES,
            config.INP_DISCOUNT_INNER_OFFSET, config.INP_DISCOUNT_OUTER_OFFSET,
            config.INP_PREMIUM_INNER_OFFSET, config.INP_PREMIUM_OUTER_OFFSET,
            point, ctf_atr)

    htf_times = htf_df["time"].values
    ctf_times = df["time"].values
    idx_map = np.searchsorted(htf_times, ctf_times, side='right') - 1
    idx_map = np.clip(idx_map, 0, len(htf_trend)-1)
    mapped_htf_trend = htf_trend[idx_map]

    if config.FILTER_HTF_ALIGNMENT:
        ctf_trend = compute_trend(df, config.TREND_ENGINE, config.PIVOT_LENGTH,
                                  config.REQUIRED_PIVOT_PAIRS, config.HOLD_LAST_TREND_ON_UNDEFINED)
    else:
        ctf_trend = np.zeros(len(df), dtype=int)

    # Qualification – the ONLY change: removed the hard‑coded neutral‑trend skip
    qualified = np.zeros(len(df), dtype=bool)
    ctf_o = df["open"].values; ctf_h = df["high"].values
    ctf_l = df["low"].values; ctf_c = df["close"].values

    if config.INP_REQUIRE_ZONE_TOUCH:
        candle_range = ctf_h - ctf_l
        valid_range = candle_range > 0
        overlap = config.INP_ZONE_OVERLAP_PERCENT / 100.0
        bull_mask = (mapped_htf_trend == 1) & valid_range & (~np.isnan(dt))
        if bull_mask.any():
            penetr_bull = np.maximum(0.0, np.minimum(ctf_h, dt) - ctf_l)
            bull_zone_ok = penetr_bull >= overlap * candle_range
            bull_mask &= bull_zone_ok
        bear_mask = (mapped_htf_trend == -1) & valid_range & (~np.isnan(pb))
        if bear_mask.any():
            penetr_bear = np.maximum(0.0, ctf_h - np.maximum(ctf_l, pb))
            bear_zone_ok = penetr_bear >= overlap * candle_range
            bear_mask &= bear_zone_ok
        zone_ok = bull_mask | bear_mask
    else:
        zone_ok = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        # ── REMOVED the filter: if mapped_htf_trend[i] == 0: continue ──
        if config.FILTER_HTF_ALIGNMENT and ctf_trend[i] != 0 and ctf_trend[i] != mapped_htf_trend[i]:
            continue
        if config.INP_REQUIRE_ZONE_TOUCH and not zone_ok[i]:
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

    # Pattern scanning
    all_hits = []
    all_counts = {}

    if config.PATTERN_LIBRARY in ("japanese", "both"):
        jp_hits = detect_japanese_hits(df, qualified, mapped_htf_trend, config.PATTERN_REQUIRE_HTF_AGREEMENT)
        for h in jp_hits: h["library"] = "japanese"
        all_hits.extend(jp_hits)
        ds_hits = detect_doji_spinning_top_hits(df, qualified, mapped_htf_trend, config.PATTERN_REQUIRE_HTF_AGREEMENT)
        all_hits.extend(ds_hits)
        all_counts["japanese"] = {}
        for pat in ["Piercing","DarkCloud","BullEngulf","BearEngulf","CustomPiercing","CustomDarkCloud","Doji","SpinningTop"]:
            all_counts["japanese"][pat] = sum(1 for h in jp_hits if h["pattern"]==pat) + \
                                          sum(1 for h in ds_hits if h["pattern"]==pat)

    if config.PATTERN_LIBRARY in ("candle_count", "both"):
        cc_hits = detect_candle_count_hits(df, qualified, mapped_htf_trend, config.PATTERN_REQUIRE_HTF_AGREEMENT)
        for h in cc_hits: h["library"] = "candle_count"
        all_hits.extend(cc_hits)
        all_counts["candle_count"] = {pat: sum(1 for h in cc_hits if h["pattern"]==pat) for pat in ["2Bull","2Bear","3Bull","3Bear"]}

    all_hits.sort(key=lambda h: h["time"])
    if config.START_DATE and have_warmup:
        all_hits = [h for h in all_hits if pd.Timestamp(h["time"]) >= pd.Timestamp(config.START_DATE)]
    if config.END_DATE and trailing_bars > 0 and report_end_ts is not None:
        all_hits = [h for h in all_hits if pd.Timestamp(h["time"]) <= report_end_ts]
    if config.PATTERN_FILTER is not None:
        all_hits = [h for h in all_hits if h["pattern"] in config.PATTERN_FILTER]

    print(f"[Indicator] pattern filter -> {config.PATTERN_FILTER or 'all'}")

    # Trade simulation
    ctf_atr_trail = precompute_atr(df, config.ATR_TRAIL_PERIOD)
    sim_desc = []
    if config.USE_ATR_TRAIL:
        sim_desc.append(f"ATR_TRAIL trigger={config.ATR_TRAIL_TRIGGER}R period={config.ATR_TRAIL_PERIOD} mult={config.ATR_TRAIL_MULT}x")
    else:
        sim_desc.append(f"RR={config.RR}")
    sim_desc.append(f"breakeven={'ON' if config.USE_BREAKEVEN else 'OFF'}")
    sim_desc.append(f"overlapping={'ON' if config.OVERLAPPING_TRADES else 'OFF'}")
    if config.USE_OPEN_SL: sim_desc.append("open-SL")
    print(f"[Scanner] simulating trades: {', '.join(sim_desc)}")

    net_r = print_detailed_trades(all_hits, df, ctf_atr_trail)
    print_performance_stats(all_hits, net_r)

    # JSON output
    if os.path.exists(config.RESULT_DIR):
        shutil.rmtree(config.RESULT_DIR)
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    out_name = config.RESULT_FILENAME if config.RESULT_FILENAME else f"{config.SYMBOL}.json"
    out_path = os.path.join(config.RESULT_DIR, out_name)
    output = {
        "symbol": config.SYMBOL,
        "ctf_timeframe": config.CTF_TIMEFRAME,
        "htf_timeframe": htf_tf,
        "trend_engine": config.TREND_ENGINE,
        "qualified_bars": int(qualified_count),
        "counts": all_counts,
        "total_hits": len(all_hits),
        "hits": all_hits,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Scanner] JSON written to: {out_path}")


if __name__ == "__main__":
    main()