"""
Offline scanner: TrendStructure + PD Zones (full port of the MQL5 indicator).

Loads an offline OHLC CSV, resamples to the mapped higher timeframe,
replays the indicator's incremental logic over the requested date range,
prints a console report, and saves a JSON report to results/.

Usage:
    python main.py
(all settings live in config.py)
"""

import os
import math
import json
import numpy as np
import pandas as pd
from datetime import datetime

import v1_config as cfg

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        # No-op decorator fallback so @njit(...) still works without numba,
        # just without the JIT speedup.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def _wrap(fn):
            return fn
        return _wrap


class WarmupError(Exception):
    """Raised when the CSV doesn't have enough history before SCAN_START
    for the trend engine / DMSI / PD zones to be past their cold-start
    state. Refusing to scan here (rather than silently producing bars
    with an undefined/neutral trend) is the point -- a scan that starts
    mid-warm-up is not wrong, just misleading."""
    pass


def fmt_time(t):
    """Format a timestamp (numpy.datetime64, pandas.Timestamp, or datetime)
    as 'YYYY-MM-DD HH:MM:SS' -- space-separated, not the ISO 'T'-separated
    default that str() produces on numpy.datetime64 / pandas.Timestamp."""
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")


# =====================================================================
# =====================================================================
#  ENGINE LOGIC
#  (Python port of the MQL5 TrendStructure + PD Zones indicator's core
#  calculation functions. Operates on plain numpy arrays representing
#  CLOSED bars only, oldest-to-newest, index 0 = oldest -- the same
#  convention the original indicator used internally after stripping
#  the still-forming bar.)
# =====================================================================
# =====================================================================

TF_MAP = {
    "M1": "M5", "M5": "M15", "M15": "M30", "M30": "H1",
    "H1": "H4", "H2": "H4", "H3": "H4", "H4": "D1", "D1": "D1",
}


def map_timeframe(tf: str) -> str:
    return TF_MAP.get(tf, "D1")


# =====================================================================
#  PIVOT DETECTION  (mirrors IsPivotHigh / IsPivotLow)
# =====================================================================
def _sliding_max(arr, window):
    """Rolling max over a centered-ish window using a monotonic deque,
    O(n) instead of O(n*window). Returns array same length as arr where
    out[i] = max(arr[i-window : i+window+1])."""
    n = len(arr)
    out = np.empty(n)
    from collections import deque
    dq = deque()  # stores indices, values decreasing
    full_window = 2 * window + 1
    for i in range(n):
        while dq and arr[dq[-1]] <= arr[i]:
            dq.pop()
        dq.append(i)
        left_edge = i - full_window + 1
        while dq[0] < left_edge:
            dq.popleft()
        if i >= full_window - 1:
            out[i - window] = arr[dq[0]]
    # Edges (first `window` and last `window` positions) are never a
    # valid pivot anyway (is_pivot_high/low require i-length>=0 and
    # i+length<n), so their value in `out` is irrelevant/unset there.
    return out


def _sliding_min(arr, window):
    n = len(arr)
    out = np.empty(n)
    from collections import deque
    dq = deque()
    full_window = 2 * window + 1
    for i in range(n):
        while dq and arr[dq[-1]] >= arr[i]:
            dq.pop()
        dq.append(i)
        left_edge = i - full_window + 1
        while dq[0] < left_edge:
            dq.popleft()
        if i >= full_window - 1:
            out[i - window] = arr[dq[0]]
    return out


def precompute_pivots(high, low, length, n):
    """Vectorized replacement for calling is_pivot_high/is_pivot_low in a
    per-bar Python loop. A bar i is a pivot high iff high[i] equals the
    max of the (2*length+1)-wide window centered on i (and is in-bounds),
    which is exactly what the strict-inequality loop in IsPivotHigh checks
    (no other bar in the window exceeds it). Same logic for pivot low with
    min. Returns two boolean arrays (is_ph, is_pl), each length n.
    NOTE: if the source window contains a duplicate of the extreme value
    at another offset, IsPivotHigh's strict '>' check would still return
    True (only a STRICTLY greater neighbor disqualifies a pivot), and the
    max-equality check here has the same behavior -- ties do not disqualify.
    """
    is_ph = np.zeros(n, dtype=bool)
    is_pl = np.zeros(n, dtype=bool)
    if n < length * 2 + 1 or length <= 0:
        return is_ph, is_pl

    win_max = _sliding_max(high, length)
    win_min = _sliding_min(low, length)

    for i in range(length, n - length):
        if high[i] >= win_max[i]:
            is_ph[i] = True
        if low[i] <= win_min[i]:
            is_pl[i] = True

    return is_ph, is_pl


def is_pivot_high(high, i, length, n):
    if i - length < 0 or i + length >= n:
        return False
    v = high[i]
    for k in range(1, length + 1):
        if high[i - k] > v or high[i + k] > v:
            return False
    return True


def is_pivot_low(low, i, length, n):
    if i - length < 0 or i + length >= n:
        return False
    v = low[i]
    for k in range(1, length + 1):
        if low[i - k] < v or low[i + k] < v:
            return False
    return True


# =====================================================================
#  ENGINE 1: PIVOT PAIRS  (AdvanceEngine_PivotPairs)
# =====================================================================
def run_pivot_pairs(high, low, n, pivot_length, required_pairs, hold_last, is_ph_arr, is_pl_arr):
    trend_out = np.zeros(n, dtype=int)
    if n < pivot_length * 2 + 5:
        return trend_out

    req_pairs = max(1, required_pairs)
    max_evaluable = n - pivot_length - 2

    hh = ll = lh = hl = 0
    trend = 0
    applied = 0
    have_pivh = have_pivl = False
    last_pivh = last_pivl = 0.0

    pending_at = []
    pending_val = []

    for i in range(0, max_evaluable + 1):
        isph = is_ph_arr[i]
        ispl = is_pl_arr[i]

        if isph:
            new_high = high[i]
            higher = True if not have_pivh else (new_high > last_pivh)
            lower = False if not have_pivh else (new_high < last_pivh)
            last_pivh = new_high
            have_pivh = True
            if higher:
                hh = min(hh + 1, req_pairs)
                lh = 0
            elif lower:
                lh = min(lh + 1, req_pairs)
                hh = 0

        if ispl:
            new_low = low[i]
            higher = True if not have_pivl else (new_low > last_pivl)
            lower = False if not have_pivl else (new_low < last_pivl)
            last_pivl = new_low
            have_pivl = True
            if higher:
                hl = min(hl + 1, req_pairs)
                ll = 0
            elif lower:
                ll = min(ll + 1, req_pairs)
                hl = 0

        if isph or ispl:
            bull_confirmed = (hh >= req_pairs) and (hl >= req_pairs)
            bear_confirmed = (lh >= req_pairs) and (ll >= req_pairs)

            new_trend = trend
            if bull_confirmed and not bear_confirmed:
                new_trend = 1
            elif bear_confirmed and not bull_confirmed:
                new_trend = -1
            elif not hold_last:
                new_trend = 0

            if new_trend != trend:
                confirm_idx = i + pivot_length
                pending_at.append(confirm_idx)
                pending_val.append(new_trend)
            trend = new_trend

    cursor = 0
    for j in range(n):
        while cursor < len(pending_at) and pending_at[cursor] == j:
            applied = pending_val[cursor]
            cursor += 1
        trend_out[j] = applied

    return trend_out


# =====================================================================
#  ENGINE 2: BOS/CHOCH  (AdvanceEngine_BosChoch)
# =====================================================================
def run_bos_choch(high, low, close, n, pivot_length, is_ph_arr, is_pl_arr):
    trend_out = np.zeros(n, dtype=int)
    if n < pivot_length * 2 + 5:
        return trend_out

    max_evaluable = n - pivot_length - 2

    confirmed_high = confirmed_low = 0.0
    have_high = have_low = False
    trend = 0

    piv_ready_at = []
    piv_ready_val = []
    piv_ready_ishigh = []
    piv_cursor = 0

    for i in range(0, max_evaluable + 1):
        isph = is_ph_arr[i]
        ispl = is_pl_arr[i]

        if isph:
            piv_ready_at.append(i + pivot_length)
            piv_ready_val.append(high[i])
            piv_ready_ishigh.append(True)
        if ispl:
            piv_ready_at.append(i + pivot_length)
            piv_ready_val.append(low[i])
            piv_ready_ishigh.append(False)

        while piv_cursor < len(piv_ready_at) and piv_ready_at[piv_cursor] == i:
            if piv_ready_ishigh[piv_cursor]:
                confirmed_high = piv_ready_val[piv_cursor]
                have_high = True
            else:
                confirmed_low = piv_ready_val[piv_cursor]
                have_low = True
            piv_cursor += 1

        trend_at_start = trend
        close_i = close[i]

        if trend_at_start <= 0 and have_high and close_i > confirmed_high:
            trend = 1
        elif trend_at_start >= 0 and have_low and close_i < confirmed_low:
            trend = -1

        trend_out[i] = trend

    if max_evaluable >= 0:
        last_val = trend_out[max_evaluable] if max_evaluable < n else 0
        for i in range(max_evaluable + 1, n):
            trend_out[i] = last_val

    return trend_out


# =====================================================================
#  ENGINE 3: CHOCH THEN BOS  (AdvanceEngine_ChochThenBos)
# =====================================================================
def run_choch_then_bos(high, low, close, n, pivot_length, hold_last, is_ph_arr, is_pl_arr):
    trend_out = np.zeros(n, dtype=int)
    if n < pivot_length * 2 + 5:
        return trend_out

    max_evaluable = n - pivot_length - 2

    confirmed_high = confirmed_low = 0.0
    have_high = have_low = False
    confirmed_trend = 0
    pending_dir = 0
    choch_level = 0.0

    piv_ready_at = []
    piv_ready_val = []
    piv_ready_ishigh = []
    piv_cursor = 0

    for i in range(0, max_evaluable + 1):
        isph = is_ph_arr[i]
        ispl = is_pl_arr[i]

        if isph:
            piv_ready_at.append(i + pivot_length)
            piv_ready_val.append(high[i])
            piv_ready_ishigh.append(True)
        if ispl:
            piv_ready_at.append(i + pivot_length)
            piv_ready_val.append(low[i])
            piv_ready_ishigh.append(False)

        while piv_cursor < len(piv_ready_at) and piv_ready_at[piv_cursor] == i:
            if piv_ready_ishigh[piv_cursor]:
                confirmed_high = piv_ready_val[piv_cursor]
                have_high = True
            else:
                confirmed_low = piv_ready_val[piv_cursor]
                have_low = True
            piv_cursor += 1

        close_i = close[i]

        if pending_dir == 0:
            if confirmed_trend <= 0 and have_high and close_i > confirmed_high:
                pending_dir = 1
                choch_level = confirmed_high
            elif confirmed_trend >= 0 and have_low and close_i < confirmed_low:
                pending_dir = -1
                choch_level = confirmed_low
        elif pending_dir == 1:
            if have_low and close_i < confirmed_low:
                pending_dir = 0
            elif close_i > choch_level:
                confirmed_trend = 1
                pending_dir = 0
        elif pending_dir == -1:
            if have_high and close_i > confirmed_high:
                pending_dir = 0
            elif close_i < choch_level:
                confirmed_trend = -1
                pending_dir = 0

        if confirmed_trend == 0 and not hold_last:
            painted = 0
        else:
            painted = confirmed_trend

        trend_out[i] = painted

    if max_evaluable >= 0:
        last_val = trend_out[max_evaluable] if max_evaluable < n else 0
        for i in range(max_evaluable + 1, n):
            trend_out[i] = last_val

    return trend_out


def compute_trend(high, low, close, n, pivot_length, required_pairs, hold_last, engine):
    is_ph_arr, is_pl_arr = precompute_pivots(high, low, pivot_length, n)
    if engine == "pivot_pairs":
        return run_pivot_pairs(high, low, n, pivot_length, required_pairs, hold_last, is_ph_arr, is_pl_arr)
    elif engine == "bos_choch":
        return run_bos_choch(high, low, close, n, pivot_length, is_ph_arr, is_pl_arr)
    else:
        return run_choch_then_bos(high, low, close, n, pivot_length, hold_last, is_ph_arr, is_pl_arr)


# =====================================================================
#  TREND AGE + PRICE POSITION  (AdvanceTrendAge)
# =====================================================================
def compute_trend_age(high, low, close, trend, n, pivot_length):
    trend_age = np.zeros(n, dtype=int)
    price_pos = np.zeros(n, dtype=float)
    price_pos_valid = np.zeros(n, dtype=bool)

    is_ph_arr, is_pl_arr = precompute_pivots(high, low, pivot_length, n)

    last_pivh = last_pivl = 0.0
    have_pivh = have_pivl = False
    last_applied_trend = 0

    for i in range(n):
        isph = is_ph_arr[i]
        ispl = is_pl_arr[i]
        if isph:
            last_pivh = high[i]
            have_pivh = True
        if ispl:
            last_pivl = low[i]
            have_pivl = True

        trend_here = trend[i]
        if i == 0 or trend_here != last_applied_trend:
            trend_age[i] = 0
        else:
            trend_age[i] = trend_age[i - 1] + 1
        last_applied_trend = trend_here

        if have_pivh and have_pivl and last_pivh > last_pivl:
            pos = (close[i] - last_pivl) / (last_pivh - last_pivl)
            pos = max(0.0, min(1.0, pos))
            price_pos[i] = pos
            price_pos_valid[i] = True
        else:
            price_pos[i] = 0.0
            price_pos_valid[i] = False

    return trend_age, price_pos, price_pos_valid


# =====================================================================
#  PIVOT LABELS  (DrawPivotLabels) -> returns list of events
# =====================================================================
def compute_pivot_labels(high, low, times, n, pivot_length):
    events = []
    if n < pivot_length * 2 + 5:
        return events
    max_evaluable = n - pivot_length - 2

    is_ph_arr, is_pl_arr = precompute_pivots(high, low, pivot_length, n)

    last_pivh = last_pivl = 0.0
    have_pivh = have_pivl = False

    for i in range(0, max_evaluable + 1):
        isph = is_ph_arr[i]
        ispl = is_pl_arr[i]
        confirm_idx = i + pivot_length

        if isph:
            new_high = high[i]
            higher = True if not have_pivh else (new_high > last_pivh)
            last_pivh = new_high
            have_pivh = True
            events.append({
                "type": "HH" if higher else "LH",
                "price": float(new_high),
                "pivot_bar_time": fmt_time(times[i]),
                "confirmed_bar_time": fmt_time(times[confirm_idx]) if confirm_idx < n else None,
            })

        if ispl:
            new_low = low[i]
            higher = True if not have_pivl else (new_low > last_pivl)
            last_pivl = new_low
            have_pivl = True
            events.append({
                "type": "HL" if higher else "LL",
                "price": float(new_low),
                "pivot_bar_time": fmt_time(times[i]),
                "confirmed_bar_time": fmt_time(times[confirm_idx]) if confirm_idx < n else None,
            })

    return events


# =====================================================================
#  CANDLE PATTERN HELPERS  (Pat* functions)
# =====================================================================
PATTERN_NAMES = ["Piercing", "DarkCloudCover", "BullishEngulfing",
                  "BearishEngulfing", "CustomPiercing", "CustomDarkCloud",
                  "MorningStar", "EveningStar",
                  "DojiConfirmedBull", "DojiConfirmedBear",
                  "SpinTopConfirmedBull", "SpinTopConfirmedBear"]
PATTERN_TAGS = ["PIER", "dark-cloud", "ENG", "ENG", "CP", "CDC",
                "MS", "ES", "DOJI+", "DOJI-", "SPIN+", "SPIN-"]
PATTERN_IS_BULL = [True, False, True, False, True, False,
                   True, False, True, False, True, False]


def resolve_enabled_patterns(names):
    """Maps a list of pattern names (from cfg.ENABLED_PATTERNS) to their
    PATTERN_NAMES indices (0-11). `names=None` means "scan all patterns" --
    returns None so compute_pattern_events applies no filter. Raises
    ValueError on an unrecognized name so a config typo fails loudly
    instead of silently scanning nothing / everything."""
    if names is None:
        return None
    unknown = [nm for nm in names if nm not in PATTERN_NAMES]
    if unknown:
        raise ValueError(
            f"ENABLED_PATTERNS contains unrecognized pattern name(s): {unknown}. "
            f"Valid names are: {PATTERN_NAMES}"
        )
    return {PATTERN_NAMES.index(nm) for nm in names}


def resolve_excluded_patterns(names):
    """Maps a list of pattern names (from cfg.EXCLUDE_PATTERNS) to their
    PATTERN_NAMES indices (0-11). `names=None` or `[]` means "exclude
    nothing" -- returns an empty set. Raises ValueError on an unrecognized
    name, same as resolve_enabled_patterns."""
    if not names:
        return set()
    unknown = [nm for nm in names if nm not in PATTERN_NAMES]
    if unknown:
        raise ValueError(
            f"EXCLUDE_PATTERNS contains unrecognized pattern name(s): {unknown}. "
            f"Valid names are: {PATTERN_NAMES}"
        )
    return {PATTERN_NAMES.index(nm) for nm in names}


def apply_pattern_exclusions(enabled_patterns, excluded_patterns):
    """Combines an ENABLED_PATTERNS set (or None = all) with an
    EXCLUDE_PATTERNS set, returning the final set of pattern indices to
    scan. Raises ValueError if the exclusion list wipes out every
    remaining pattern, so a contradictory config (e.g. excluding
    everything ENABLED_PATTERNS included) fails loudly."""
    if not excluded_patterns:
        return enabled_patterns
    base = set(range(len(PATTERN_NAMES))) if enabled_patterns is None else enabled_patterns
    result = base - excluded_patterns
    if not result:
        raise ValueError(
            "EXCLUDE_PATTERNS removes every pattern that would otherwise be "
            "scanned (check for overlap with ENABLED_PATTERNS)."
        )
    return result


def _body(o, c, i):
    return abs(c[i] - o[i])


def _range(h, l, i):
    return h[i] - l[i]


def _is_bull(o, c, i):
    return c[i] > o[i]


def _is_bear(o, c, i):
    return c[i] < o[i]


def _body_ratio_at_least(o, h, l, c, i, ratio):
    r = _range(h, l, i)
    if r <= 0:
        return False
    return (_body(o, c, i) / r) >= ratio


def _body_ratio_at_most(o, h, l, c, i, ratio):
    r = _range(h, l, i)
    if r <= 0:
        return True
    return (_body(o, c, i) / r) <= ratio


def _is_doji(o, h, l, c, i, doji_ratio):
    return _body_ratio_at_most(o, h, l, c, i, doji_ratio)


def _is_spinning_top(o, h, l, c, i, doji_ratio, spin_ratio):
    r = _range(h, l, i)
    if r <= 0:
        return False
    effective = max(spin_ratio, doji_ratio)
    body_ratio = _body(o, c, i) / r
    return (body_ratio > doji_ratio) and (body_ratio <= effective)


def _is_doji_or_spin(o, h, l, c, i, doji_ratio, spin_ratio):
    return _is_doji(o, h, l, c, i, doji_ratio) or _is_spinning_top(o, h, l, c, i, doji_ratio, spin_ratio)


def _is_piercing(o, h, l, c, i, long_ratio):
    if i < 1:
        return False
    if not _is_bear(o, c, i - 1):
        return False
    if not _body_ratio_at_least(o, h, l, c, i - 1, long_ratio):
        return False
    if not _is_bull(o, c, i):
        return False
    if o[i] >= l[i - 1]:
        return False
    if c[i] <= c[i - 1]:
        return False
    return True


def _is_dark_cloud(o, h, l, c, i, long_ratio):
    if i < 1:
        return False
    if not _is_bull(o, c, i - 1):
        return False
    if not _body_ratio_at_least(o, h, l, c, i - 1, long_ratio):
        return False
    if not _is_bear(o, c, i):
        return False
    if o[i] <= h[i - 1]:
        return False
    if c[i] >= c[i - 1]:
        return False
    return True


def _is_bull_engulf(o, h, l, c, i, doji_ratio, spin_ratio):
    if i < 1:
        return False
    if not _is_bear(o, c, i - 1):
        return False
    if not _is_bull(o, c, i):
        return False
    if _is_doji_or_spin(o, h, l, c, i, doji_ratio, spin_ratio):
        return False
    if o[i] > c[i - 1]:
        return False
    if c[i] <= o[i - 1]:
        return False
    if l[i] >= l[i - 1]:
        return False
    if h[i] <= h[i - 1]:
        return False
    return True


def _is_bear_engulf(o, h, l, c, i, doji_ratio, spin_ratio):
    if i < 1:
        return False
    if not _is_bull(o, c, i - 1):
        return False
    if not _is_bear(o, c, i):
        return False
    if _is_doji_or_spin(o, h, l, c, i, doji_ratio, spin_ratio):
        return False
    if o[i] < c[i - 1]:
        return False
    if c[i] >= o[i - 1]:
        return False
    if h[i] <= h[i - 1]:
        return False
    if l[i] >= l[i - 1]:
        return False
    return True


def _is_custom_piercing(o, h, l, c, i, doji_ratio, spin_ratio, long_ratio):
    # Mirrors MQL5 PatIsCustomPiercing exactly: previous candle only needs
    # to be a plain bearish candle (no doji/spin allowance, no long-body
    # requirement) -- there is no additional gate on candle i-1's shape.
    if i < 1:
        return False
    if not _is_bear(o, c, i - 1):
        return False
    if not _is_bull(o, c, i):
        return False
    if _is_doji_or_spin(o, h, l, c, i, doji_ratio, spin_ratio):
        return False
    if l[i] >= l[i - 1]:
        return False
    if c[i] <= c[i - 1]:
        return False
    return True


def _is_custom_dark_cloud(o, h, l, c, i, doji_ratio, spin_ratio, long_ratio):
    # Mirrors MQL5 PatIsCustomDarkCloud exactly: previous candle only needs
    # to be a plain bullish candle (no doji/spin allowance, no long-body
    # requirement) -- there is no additional gate on candle i-1's shape.
    if i < 1:
        return False
    if not _is_bull(o, c, i - 1):
        return False
    if not _is_bear(o, c, i):
        return False
    if _is_doji_or_spin(o, h, l, c, i, doji_ratio, spin_ratio):
        return False
    if h[i] <= h[i - 1]:
        return False
    if c[i] >= c[i - 1]:
        return False
    return True


# --- Morning Star (bullish, 3-candle): i-2 long bearish, i-1 small body
# (no gap required, just small body), i closes past the midpoint of
# candle i-2's body. Mirrors MQL5's PatIsMorningStar. ---
def _is_morning_star(o, h, l, c, i, doji_ratio, spin_ratio, long_ratio):
    if i < 2:
        return False
    if not _is_bear(o, c, i - 2):
        return False
    if not _body_ratio_at_least(o, h, l, c, i - 2, long_ratio):
        return False
    if not _is_doji_or_spin(o, h, l, c, i - 1, doji_ratio, spin_ratio):
        return False
    if not _is_bull(o, c, i):
        return False
    midpoint = (o[i - 2] + c[i - 2]) / 2.0
    if c[i] <= midpoint:
        return False
    return True


# --- Evening Star (bearish, 3-candle mirror). Mirrors PatIsEveningStar. ---
def _is_evening_star(o, h, l, c, i, doji_ratio, spin_ratio, long_ratio):
    if i < 2:
        return False
    if not _is_bull(o, c, i - 2):
        return False
    if not _body_ratio_at_least(o, h, l, c, i - 2, long_ratio):
        return False
    if not _is_doji_or_spin(o, h, l, c, i - 1, doji_ratio, spin_ratio):
        return False
    if not _is_bear(o, c, i):
        return False
    midpoint = (o[i - 2] + c[i - 2]) / 2.0
    if c[i] >= midpoint:
        return False
    return True


# --- Doji / Spinning Top "strong break" confirmations (2-candle):
# prior candle (i-1) is a Doji or Spinning Top shape; confirmation
# candle (i) must close beyond the prior candle's high (bull) or
# low (bear). Mirrors PatIsDojiConfirmedBull/Bear and
# PatIsSpinTopConfirmedBull/Bear. ---
def _is_doji_confirmed_bull(o, h, l, c, i, doji_ratio):
    if i < 1:
        return False
    if not _is_doji(o, h, l, c, i - 1, doji_ratio):
        return False
    if c[i] <= h[i - 1]:
        return False
    return True


def _is_doji_confirmed_bear(o, h, l, c, i, doji_ratio):
    if i < 1:
        return False
    if not _is_doji(o, h, l, c, i - 1, doji_ratio):
        return False
    if c[i] >= l[i - 1]:
        return False
    return True


def _is_spintop_confirmed_bull(o, h, l, c, i, doji_ratio, spin_ratio):
    if i < 1:
        return False
    if not _is_spinning_top(o, h, l, c, i - 1, doji_ratio, spin_ratio):
        return False
    if c[i] <= h[i - 1]:
        return False
    return True


def _is_spintop_confirmed_bear(o, h, l, c, i, doji_ratio, spin_ratio):
    if i < 1:
        return False
    if not _is_spinning_top(o, h, l, c, i - 1, doji_ratio, spin_ratio):
        return False
    if c[i] >= l[i - 1]:
        return False
    return True


def evaluate_patterns_at(o, h, l, c, i, doji_ratio, spin_ratio, long_ratio):
    """Returns list of pattern indices (0-11) hit at bar i, honoring the
    original's strict priority: Piercing/DCC > Engulfing > Custom CP/CDC >
    Morning/Evening Star > Doji/SpinningTop confirmed (lowest, most generic)."""
    if _is_piercing(o, h, l, c, i, long_ratio):
        return [0]
    if _is_dark_cloud(o, h, l, c, i, long_ratio):
        return [1]

    hits = []
    bull_engulf = _is_bull_engulf(o, h, l, c, i, doji_ratio, spin_ratio)
    bear_engulf = _is_bear_engulf(o, h, l, c, i, doji_ratio, spin_ratio)
    if bull_engulf:
        hits.append(2)
    elif bear_engulf:
        hits.append(3)

    if not bull_engulf and not bear_engulf:
        if _is_custom_piercing(o, h, l, c, i, doji_ratio, spin_ratio, long_ratio):
            hits.append(4)
        elif _is_custom_dark_cloud(o, h, l, c, i, doji_ratio, spin_ratio, long_ratio):
            hits.append(5)

    if not hits:
        if _is_morning_star(o, h, l, c, i, doji_ratio, spin_ratio, long_ratio):
            hits.append(6)
        elif _is_evening_star(o, h, l, c, i, doji_ratio, spin_ratio, long_ratio):
            hits.append(7)

    if not hits:
        if _is_doji_confirmed_bull(o, h, l, c, i, doji_ratio):
            hits.append(8)
        elif _is_doji_confirmed_bear(o, h, l, c, i, doji_ratio):
            hits.append(9)
        elif _is_spintop_confirmed_bull(o, h, l, c, i, doji_ratio, spin_ratio):
            hits.append(10)
        elif _is_spintop_confirmed_bear(o, h, l, c, i, doji_ratio, spin_ratio):
            hits.append(11)

    return hits


# =====================================================================
#  HTF TREND LOOKUP (binary search, mirrors HtfTrendAtTime)
# =====================================================================
def htf_trend_at_time(htf_close_times, htf_trend, t):
    """Returns the trend of the last HTF bar that has FULLY CLOSED by time t.
    `htf_close_times` must be each HTF bar's close time (open time + bar
    duration), not its open time -- using open time here would let a CTF
    bar see the trend of its own still-forming HTF parent bar (look-ahead
    bias), since this offline scanner resamples the whole HTF series
    upfront and has no live/forming-bar distinction the way MT5's
    CopyRates(..., shift=1, ...) live-cache exclusion does."""
    lo, hi, ans = 0, len(htf_close_times) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if htf_close_times[mid] <= t:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if ans < 0 or htf_close_times[ans] > t:
        return 0
    return htf_trend[ans]


def pattern_filters_allow(ctf_times, i, span, htf_close_times, htf_trend, is_bull, require_agreement):
    """HTF-agreement filter: agreement only needs to hold on ANY one of the
    candles spanned by the pattern -- not all of them. Mirrors the latest
    MQL5 PatternFiltersAllow behavior (span = 2 for most patterns, 3 for
    Morning/Evening Star). `htf_close_times` must be HTF bar CLOSE times
    (see htf_trend_at_time)."""
    if not require_agreement:
        return True
    for k in range(span):
        idx = i - k
        if idx < 0:
            continue
        htf_trend_here = htf_trend_at_time(htf_close_times, htf_trend, ctf_times[idx])
        pass_htf = (htf_trend_here == 1) if is_bull else (htf_trend_here == -1)
        if pass_htf:
            return True
    return False


def pattern_candle_count(pat_idx):
    """Mirrors PatternCandleCount: Morning/Evening Star (indices 6, 7) span
    3 candles; every other pattern spans 2."""
    return 3 if pat_idx in (6, 7) else 2


def compute_pattern_events(ctf_times, o, h, l, c, color_buf, n,
                            doji_ratio, spin_ratio, long_ratio,
                            require_htf_agreement, htf_close_times, htf_trend,
                            enabled_pattern_indices=None):
    """Mirrors MQL5 AdvancePatternLabels exactly: a bar is only skipped when
    it's painted neutral(0) or htf_disagree(4). Any pattern hit at an
    eligible bar is drawn using that pattern's own bull/bear color --
    there is no additional requirement that the pattern's direction match
    which specific color (1=bull vs 2=bear) the bar was painted.
    PatternFiltersAllow (raw HTF trend agreement) is still the only extra
    gate applied on top.

    `enabled_pattern_indices`: optional set/list of PATTERN_NAMES indices
    (0-11) to restrict scanning to. None (default) scans all 12 -- matches
    the original indicator, which has no such filter. This is purely a
    scanner-side convenience; MQL5 has no equivalent switch.

    Each event doubles as a trade `hit` for simulate_trade(): includes
    bar_index, direction ("bull"/"bear"), sl_extreme (signal bar's low for
    bull, high for bear), and time (alias of bar_time) for report printing.
    """
    enabled = set(range(len(PATTERN_NAMES))) if enabled_pattern_indices is None else set(enabled_pattern_indices)
    events = []
    for i in range(2, n):
        if i >= len(color_buf) or color_buf[i] in (0, 4):
            continue
        hits = evaluate_patterns_at(o, h, l, c, i, doji_ratio, spin_ratio, long_ratio)
        if not hits:
            continue
        for p in hits:
            if p not in enabled:
                continue
            is_bull = PATTERN_IS_BULL[p]
            span = pattern_candle_count(p)
            if not pattern_filters_allow(ctf_times, i, span, htf_close_times, htf_trend, is_bull, require_htf_agreement):
                continue
            sl_extreme = float(l[i]) if is_bull else float(h[i])
            events.append({
                "pattern": PATTERN_NAMES[p],
                "tag": PATTERN_TAGS[p],
                "bull": is_bull,
                "direction": "bull" if is_bull else "bear",
                "bar_index": i,
                "sl_extreme": sl_extreme,
                "bar_time": fmt_time(ctf_times[i]),
                "prev_bar_time": fmt_time(ctf_times[i - 1]),
                "time": fmt_time(ctf_times[i]),
            })
    return events


# =====================================================================
#  PD ZONES  (PDZ_AdvanceHTFZones / PDZ_RunCalculate core math)
# =====================================================================
def calc_atr(high, low, close, i, period):
    if i < period or period <= 0:
        return 0.0
    total = 0.0
    for j in range(period):
        tr = high[i - j] - low[i - j]
        if i - j - 1 >= 0:
            tr2 = abs(high[i - j] - close[i - j - 1])
            tr3 = abs(low[i - j] - close[i - j - 1])
            tr = max(tr, tr2, tr3)
        total += tr
    return total / period


def _vectorized_atr_series(high, low, close, period):
    """Vectorized equivalent of calling calc_atr(high, low, close, i, period)
    for every i: True Range then a trailing rolling mean over `period` bars
    (window = [i-period+1, i], matching calc_atr's backward-looking sum/period
    exactly -- verified bit-for-bit against the per-bar loop). Returns an
    array where index i < period holds 0.0, matching calc_atr's `if i < period:
    return 0.0` guard."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    if n > 1:
        prev_close = close[:-1]
        h2 = high[1:]
        l2 = low[1:]
        tr[1:] = np.maximum(h2 - l2, np.maximum(np.abs(h2 - prev_close), np.abs(l2 - prev_close)))
    atr = pd.Series(tr).rolling(period).mean().values
    atr = np.where(np.arange(n) < period, 0.0, atr)
    atr = np.nan_to_num(atr, nan=0.0)
    return atr


def compute_zones(high, low, close, n, lookback, use_atr, atr_period, atr_mult,
                   use_manual, disc_inner, disc_outer, prem_inner, prem_outer,
                   point_size, buffer_points):
    eq = np.zeros(n)
    premium_top = np.zeros(n)
    premium_bot = np.zeros(n)
    discount_top = np.zeros(n)
    discount_bot = np.zeros(n)
    valid = np.zeros(n, dtype=bool)

    if n < lookback:
        return {
            "eq": eq, "premium_top": premium_top, "premium_bot": premium_bot,
            "discount_top": discount_top, "discount_bot": discount_bot, "valid": valid,
        }

    # Trailing rolling max/min over `lookback` bars (window [i-lookback+1, i],
    # clipped at 0 for early bars), matching the original's
    # high[max(0,i-lookback+1):i+1].max() slice exactly.
    highest_roll = pd.Series(high).rolling(lookback, min_periods=1).max().values
    lowest_roll = pd.Series(low).rolling(lookback, min_periods=1).min().values
    midpoint_arr = (highest_roll + lowest_roll) / 2.0

    if not use_manual and use_atr:
        atr_arr = _vectorized_atr_series(high, low, close, atr_period)
    else:
        atr_arr = None

    for i in range(lookback - 1, n):
        midpoint = midpoint_arr[i]

        if use_manual:
            d_top = midpoint - disc_inner
            d_bot = midpoint - disc_outer
            p_bot = midpoint + prem_inner
            p_top = midpoint + prem_outer
        else:
            if use_atr and i >= atr_period:
                buf = atr_arr[i] * atr_mult
                if buf <= 0:
                    buf = buffer_points * point_size
            else:
                buf = buffer_points * point_size
            p_bot = midpoint + buf
            p_top = midpoint + buf * 3
            d_top = midpoint - buf
            d_bot = midpoint - buf * 3

        eq[i] = midpoint
        premium_top[i] = p_top
        premium_bot[i] = p_bot
        discount_top[i] = d_top
        discount_bot[i] = d_bot
        valid[i] = True

    return {
        "eq": eq, "premium_top": premium_top, "premium_bot": premium_bot,
        "discount_top": discount_top, "discount_bot": discount_bot, "valid": valid,
    }


def map_ctf_zones_from_htf(ctf_times, htf_close_times, htf_zones, n):
    """For each CTF bar, find the last HTF bar that has FULLY CLOSED
    at-or-before it (a monotone 'as-of join' on CLOSE time) and copy its
    zone values across if that HTF bar is valid. Vectorized via
    np.searchsorted -- equivalent to the original two-pointer walk: 'the
    last HTF bar with htf_close_times[use_idx] <= t', same <= tie
    behavior on both sides, same no-HTF-yet (t before first HTF bar
    closes) case.

    IMPORTANT: `htf_close_times` must be each HTF bar's CLOSE time (open
    time + bar duration), not its open time. Joining on open time would
    let a CTF bar see the zone of its own still-forming HTF parent bar --
    a look-ahead bias, since this offline scanner resamples the whole HTF
    series upfront (no live/forming-bar distinction), unlike MT5's
    CopyRates(..., shift=1, ...) which excludes the live bar from its
    HTF cache."""
    out_eq = np.full(n, np.nan)
    out_pt = np.full(n, np.nan)
    out_pb = np.full(n, np.nan)
    out_dt = np.full(n, np.nan)
    out_db = np.full(n, np.nan)

    last_valid_idx = len(htf_close_times) - 1
    if last_valid_idx < 0:
        return out_eq, out_pt, out_pb, out_dt, out_db

    # side="right": first HTF index strictly greater than t; minus 1 gives
    # the last HTF index with htf_close_times[idx] <= t. -1 means no such
    # (fully closed) bar yet.
    use_idx = np.searchsorted(htf_close_times, ctf_times, side="right") - 1
    has_htf = use_idx >= 0
    use_idx_clamped = np.clip(use_idx, 0, last_valid_idx)

    valid_arr = np.asarray(htf_zones["valid"])
    is_valid = has_htf & valid_arr[use_idx_clamped]

    out_eq[is_valid] = np.asarray(htf_zones["eq"])[use_idx_clamped][is_valid]
    out_pt[is_valid] = np.asarray(htf_zones["premium_top"])[use_idx_clamped][is_valid]
    out_pb[is_valid] = np.asarray(htf_zones["premium_bot"])[use_idx_clamped][is_valid]
    out_dt[is_valid] = np.asarray(htf_zones["discount_top"])[use_idx_clamped][is_valid]
    out_db[is_valid] = np.asarray(htf_zones["discount_bot"])[use_idx_clamped][is_valid]

    return out_eq, out_pt, out_pb, out_dt, out_db


# =====================================================================
#  ZONE-TOUCH FILTER  (mirrors the InpRequireZoneTouch block)
# =====================================================================
def zone_touch_allows(close_i, low_i, high_i, zone_low, zone_high, overlap_threshold):
    if math.isnan(zone_low) or math.isnan(zone_high):
        return False
    if zone_low <= close_i <= zone_high:
        return True
    overlap = max(0.0, min(high_i, zone_high) - max(low_i, zone_low))
    rng = high_i - low_i
    if overlap > 0 and rng > 0 and overlap >= overlap_threshold * rng:
        return True
    return False


# =====================================================================
#  DMSI (DYNAMIC MARKET STRUCTURE INDEX) -- MERGED REGIME FILTER
#  Mirrors the merged MQL5 DMSI_ComputeAt / DMSI_RegimeAt block.
#  Forward-indexed (index 0 = oldest), matching this file's convention
#  (the original standalone DMSI indicator was series-indexed).
#
#  Regime definitions:
#    0 = Trending  (DMSI >= adaptive trend percentile level)
#    1 = Transition
#    2 = Ranging   (DMSI <= adaptive range percentile level)
#
#  Only regime 0 (Trending) passes the filter; Ranging/Transition both
#  downgrade a candle to neutral.
# =====================================================================
@njit(cache=True)
def _compute_dmsi_core(close, n, lookback, percentile_period,
                        trend_pct, range_pct, sc_range, slowest,
                        min_bars, seed_idx,
                        dmsi_raw, dmsi_value, trend_level, range_level):
    """JIT-compiled inner loop of compute_dmsi. Sequential (dmsi_value[i]
    depends on dmsi_value[i-1]), so it can't be vectorized with numpy --
    same algorithm as the original pure-Python version, just compiled.
    Arrays are pre-allocated and passed in; mutated in place."""
    nd = float(lookback)
    sum_x = nd * (nd - 1.0) * 0.5
    sum_x2 = (nd - 1.0) * nd * (2.0 * nd - 1.0) / 6.0
    denom = nd * sum_x2 - sum_x * sum_x
    inv_n = 1.0 / nd

    for i in range(min_bars, n):
        # 1. Kaufman Efficiency Ratio (bars i-lookback .. i)
        net_change = abs(close[i] - close[i - lookback])
        gross_change = 0.0
        for j in range(lookback):
            gross_change += abs(close[i - j] - close[i - j - 1])

        er = 0.0
        if gross_change > 1e-12:
            er = net_change / gross_change
            if er > 1.0:
                er = 1.0

        # 2. Linear Regression R^2 over trailing window (i-lookback+1 .. i)
        sum_y = 0.0
        sum_xy = 0.0
        sum_y2 = 0.0
        for j in range(lookback):
            y = close[i - lookback + 1 + j]
            x = float(j)
            sum_y += y
            sum_xy += x * y
            sum_y2 += y * y

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

        r_squared = 1.0 - sse / sst
        if r_squared < 0.0:
            r_squared = 0.0
        elif r_squared > 1.0:
            r_squared = 1.0

        # 3. Fusion: ER + R^2, scaled to 0-100
        raw_scaled = (er + r_squared) * 50.0
        dmsi_raw[i] = raw_scaled

        # 4. Adaptive Kaufman smoothing (recursive on the previous bar, i-1)
        sc = er * sc_range + slowest
        sc = sc * sc

        if i <= seed_idx:
            dmsi_value[i] = raw_scaled  # seed at the oldest computable bar
        else:
            dmsi_value[i] = dmsi_value[i - 1] + sc * (raw_scaled - dmsi_value[i - 1])

        # 5. Dynamic percentile levels over the trailing dmsi_value window
        available = i + 1
        window = min(available, percentile_period)

        if window < 5:
            trend_level[i] = 70.0
            range_level[i] = 30.0
        else:
            temp = dmsi_value[i - window + 1: i + 1].copy()
            temp.sort()

            idx_trend = int(trend_pct * (window - 1))
            idx_range = int(range_pct * (window - 1))
            if idx_trend >= window:
                idx_trend = window - 1
            if idx_range >= window:
                idx_range = window - 1

            trend_level[i] = temp[idx_trend]
            range_level[i] = temp[idx_range]


def compute_dmsi(high, low, close, n, lookback, percentile_period,
                  trend_percentile, range_percentile, fast_smooth, slow_smooth):
    """Returns (dmsi_value, trend_level, range_level) arrays, each length n.
    Indices before the warm-up point (lookback*2 + 10) are left at 0.0 /
    NaN and should not be read -- mirrors dmsi_min_bars gating in MQL5.
    Numerically identical to the original pure-Python loop; the hot inner
    loop is JIT-compiled via numba when available (falls back to plain
    Python, same logic, if numba isn't installed)."""
    dmsi_raw = np.zeros(n, dtype=float)
    dmsi_value = np.zeros(n, dtype=float)
    trend_level = np.full(n, np.nan, dtype=float)
    range_level = np.full(n, np.nan, dtype=float)

    min_bars = lookback * 2 + 10
    if n <= min_bars:
        return dmsi_value, trend_level, range_level

    fastest = 2.0 / (fast_smooth + 1.0)
    slowest = 2.0 / (slow_smooth + 1.0)
    sc_range = fastest - slowest

    trend_pct = trend_percentile / 100.0
    range_pct = range_percentile / 100.0

    seed_idx = min_bars

    close_arr = np.ascontiguousarray(close, dtype=np.float64)

    _compute_dmsi_core(
        close_arr, n, lookback, percentile_period,
        trend_pct, range_pct, sc_range, slowest,
        min_bars, seed_idx,
        dmsi_raw, dmsi_value, trend_level, range_level,
    )

    return dmsi_value, trend_level, range_level


def dmsi_regime_at(dmsi_value, trend_level, range_level, i):
    """Returns 0=Trending, 1=Transition, 2=Ranging at forward-index i."""
    if dmsi_value[i] >= trend_level[i]:
        return 0
    if dmsi_value[i] <= range_level[i]:
        return 2
    return 1


# =====================================================================
#  ATR (Wilder-style rolling-mean TR) -- used only for ATR trailing stops
#  in trade simulation. NOTE: distinct from calc_atr() above, which is
#  the indicator's own simple-average ATR used for PD zone buffers.
# =====================================================================
def precompute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    """Return array of same length as df, NaN where i < period-1.
    Vectorized (same True Range + rolling-mean shape as
    _vectorized_atr_series) instead of a pure-Python per-bar loop."""
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    n = len(df)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    if n > 1:
        prev_close = close[:-1]
        h2 = high[1:]
        l2 = low[1:]
        tr[1:] = np.maximum(h2 - l2, np.maximum(np.abs(h2 - prev_close), np.abs(l2 - prev_close)))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1:] = pd.Series(tr).rolling(period).mean().values[period - 1:]
    return atr


# =====================================================================
#  TRADE SIMULATION  (per-hit forward walk: SL / TP / breakeven / ATR trail)
# =====================================================================
@njit(cache=True)
def _simulate_trade_core(entry_idx, entry, is_bull, sl, n, rr,
                          use_breakeven, be_trigger, be_buffer,
                          atr_trigger, atr_mult,
                          use_open_sl, h, l, c, atr_array,
                          fast_atr_array, step2_r, step2_mult, step3_r, step3_mult,
                          trail_type, is_ph, is_pl, swing_buffer_mult,
                          ratchet_trig, ratchet_lock, n_ratchet_steps,
                          psar_af_start, psar_af_step, psar_af_max,
                          time_bars_full_tighten, time_mult_start, time_mult_end,
                          dmsi_value, dmsi_trend_level, dmsi_range_level,
                          dmsi_wide_mult, dmsi_tight_mult):
    """JIT-compiled forward-walk loop for a single trade -- the hot path
    of the whole scanner. Called once per pattern hit; each call can
    scan thousands of bars, so compiling it (vs. pure Python) is the
    single biggest speed win available in this file. Returns a
    fixed-shape tuple (numba can't return heterogeneous dicts);
    simulate_trade() below unpacks it back into the original dict shape.

    outcome codes: 0=invalid 1=breakeven 2=sl 3=trail 4=tp 5=open

    trail_type selects which trailing algorithm drives current_sl once
    armed (arming itself is always "max_favorable_r >= atr_trigger",
    shared across all types so TRAIL_TYPE comparisons stay apples-to-
    apples). Every type still only ever moves current_sl favorably
    (ratchets), consistent with the rest of this function's exit logic:

      0 = OFF            -- no trailing; TP (rr) is the only exit (plus
                             breakeven/original SL). Kept for A/B baseline.
      1 = STEPPED_ATR     -- fast-ATR distance, multiplier tightens in two
                             steps as max_favorable_r extends (step2/step3).
      2 = CHANDELIER_ATR  -- distance anchored to the highest-high (bull) /
                             lowest-low (bear) reached so far in the trade,
                             not this bar's high/low, so a single spike bar
                             can't create a level threatened by the very
                             next ordinary pullback bar.
      3 = STRUCTURE_SWING -- trail behind the last CONFIRMED pivot low
                             (bull) / pivot high (bear) from precompute_pivots,
                             plus a small fast-ATR buffer. Room scales with
                             the market's own swing structure instead of a
                             volatility distance.
      4 = R_RATCHET       -- deterministic, no ATR at all: current_sl steps
                             to ratchet_lock[k]*risk (in R, added to entry)
                             the first time max_favorable_r >= ratchet_trig[k],
                             for k = 0..n_ratchet_steps-1.
      5 = PARABOLIC       -- SAR-style: acceleration factor starts at
                             psar_af_start and increases by psar_af_step
                             (capped at psar_af_max) every time a new
                             favorable extreme is made; trail distance from
                             that extreme shrinks as af grows, so the trail
                             accelerates continuously instead of in steps.
      6 = TIME_BASED      -- multiplier on fast_atr_array shrinks linearly
                             from time_mult_start to time_mult_end as bars-
                             since-entry goes from 0 to time_bars_full_tighten
                             (then holds at time_mult_end), independent of
                             price/ATR magnitude.
      7 = DMSI_REGIME     -- fast-ATR multiplier is dmsi_wide_mult while the
                             DMSI regime at the current bar is "trending",
                             and snaps to the tighter dmsi_tight_mult the
                             moment it's not (transition or ranging) --
                             ties the trail to the same trend/range read
                             used for entries instead of a generic distance.
      8 = RATCHET_ATR_HYBRID -- deterministic R_RATCHET floor (locks in
                             ratchet_lock[k]*risk the first time
                             max_favorable_r >= ratchet_trig[k], exactly as
                             type 4) combined with the STEPPED_ATR distance
                             trail (type 1) running on top of it. Each bar,
                             current_sl becomes whichever of the two
                             candidates is more favorable -- so a sharp
                             reversal can never take profit back below the
                             last locked milestone, while the ATR trail
                             still tightens further between milestones.
    """
    risk = abs(entry - sl)
    if risk <= 0.0:
        return (0, 0.0, 0.0, np.nan, 0.0, -1, False, False)

    trail_enabled = trail_type != 0   # TRAIL_TYPE == "OFF" -> fixed TP only, no trailing

    if not trail_enabled:
        tp = entry + rr * risk if is_bull else entry - rr * risk
        has_tp = True
    else:
        tp = 0.0
        has_tp = False

    be_stop = entry + be_buffer * risk if is_bull else entry - be_buffer * risk
    be_trigger_price = entry + be_trigger * risk if is_bull else entry - be_trigger * risk

    current_sl = sl
    breakeven_armed = False
    trail_armed = False
    max_favorable_r = 0.0
    sl_moved = False

    # -- state used only by specific trail types --
    chandelier_extreme = entry            # type 2: best high/low seen since entry
    last_swing_level = np.nan             # type 3: most recent confirmed opposite-side pivot
    ratchet_next_step = 0                 # type 4: index into ratchet_trig/ratchet_lock
    psar_af = psar_af_start               # type 5: current acceleration factor
    psar_extreme = entry                  # type 5: best favorable extreme seen since arming

    for idx in range(entry_idx + 1, n):
        bar_high = h[idx]
        bar_low = l[idx]

        fav_extreme = bar_high if is_bull else bar_low
        bar_fav_r = (fav_extreme - entry) / risk if is_bull else (entry - fav_extreme) / risk
        if bar_fav_r > max_favorable_r:
            max_favorable_r = bar_fav_r

        if is_bull and bar_high > chandelier_extreme:
            chandelier_extreme = bar_high
        elif not is_bull and bar_low < chandelier_extreme:
            chandelier_extreme = bar_low

        # track most recent CONFIRMED opposite-side pivot for structure trail
        # (is_ph/is_pl[idx] True means idx itself is a confirmed pivot bar --
        # precompute_pivots already requires `length` bars of right-side
        # confirmation, so no repainting/lookahead here)
        if trail_type == 3:
            if is_bull and is_pl[idx]:
                last_swing_level = bar_low
            elif not is_bull and is_ph[idx]:
                last_swing_level = bar_high

        if trail_enabled and not trail_armed and max_favorable_r >= atr_trigger:
            trail_armed = True
            psar_extreme = fav_extreme

        if trail_armed:
            if trail_type == 0:
                pass  # OFF: arming still recorded (trail_active reporting) but current_sl never moves

            elif trail_type == 1:
                # STEPPED_ATR
                if max_favorable_r >= step3_r:
                    active_mult = step3_mult
                elif max_favorable_r >= step2_r:
                    active_mult = step2_mult
                else:
                    active_mult = atr_mult
                atr_val = fast_atr_array[idx]
                if not np.isnan(atr_val):
                    candidate = chandelier_extreme - active_mult * atr_val if is_bull \
                        else chandelier_extreme + active_mult * atr_val
                    if (is_bull and candidate > current_sl) or (not is_bull and candidate < current_sl):
                        current_sl = candidate
                        sl_moved = True

            elif trail_type == 2:
                # CHANDELIER_ATR -- distance from the best extreme seen so
                # far (chandelier_extreme), not this bar's high/low
                atr_val = fast_atr_array[idx]
                if not np.isnan(atr_val):
                    candidate = chandelier_extreme - atr_mult * atr_val if is_bull \
                        else chandelier_extreme + atr_mult * atr_val
                    if (is_bull and candidate > current_sl) or (not is_bull and candidate < current_sl):
                        current_sl = candidate
                        sl_moved = True

            elif trail_type == 3:
                # STRUCTURE_SWING -- last confirmed opposite pivot + small buffer
                if not np.isnan(last_swing_level):
                    atr_val = fast_atr_array[idx]
                    buf = swing_buffer_mult * atr_val if not np.isnan(atr_val) else 0.0
                    candidate = last_swing_level - buf if is_bull else last_swing_level + buf
                    if (is_bull and candidate > current_sl) or (not is_bull and candidate < current_sl):
                        current_sl = candidate
                        sl_moved = True

            elif trail_type == 4:
                # R_RATCHET -- deterministic step to entry +/- ratchet_lock[k]*risk
                while ratchet_next_step < n_ratchet_steps and \
                        max_favorable_r >= ratchet_trig[ratchet_next_step]:
                    lock_r = ratchet_lock[ratchet_next_step]
                    candidate = entry + lock_r * risk if is_bull else entry - lock_r * risk
                    if (is_bull and candidate > current_sl) or (not is_bull and candidate < current_sl):
                        current_sl = candidate
                        sl_moved = True
                    ratchet_next_step += 1

            elif trail_type == 5:
                # PARABOLIC -- SAR-style continuous acceleration
                made_new_extreme = (is_bull and fav_extreme > psar_extreme) or \
                                    (not is_bull and fav_extreme < psar_extreme)
                if made_new_extreme:
                    psar_extreme = fav_extreme
                    if psar_af < psar_af_max:
                        psar_af = psar_af + psar_af_step
                        if psar_af > psar_af_max:
                            psar_af = psar_af_max
                atr_val = fast_atr_array[idx]
                if not np.isnan(atr_val):
                    dist = (1.0 / psar_af) * atr_val if psar_af > 0.0 else atr_val
                    candidate = psar_extreme - dist if is_bull else psar_extreme + dist
                    if (is_bull and candidate > current_sl) or (not is_bull and candidate < current_sl):
                        current_sl = candidate
                        sl_moved = True

            elif trail_type == 6:
                # TIME_BASED -- multiplier shrinks linearly with bars-since-entry
                bars_since_entry = idx - entry_idx
                if time_bars_full_tighten > 0:
                    frac = bars_since_entry / time_bars_full_tighten
                    if frac > 1.0:
                        frac = 1.0
                else:
                    frac = 1.0
                active_mult = time_mult_start + (time_mult_end - time_mult_start) * frac
                atr_val = fast_atr_array[idx]
                if not np.isnan(atr_val):
                    candidate = chandelier_extreme - active_mult * atr_val if is_bull \
                        else chandelier_extreme + active_mult * atr_val
                    if (is_bull and candidate > current_sl) or (not is_bull and candidate < current_sl):
                        current_sl = candidate
                        sl_moved = True

            elif trail_type == 7:
                # DMSI_REGIME -- wide mult while trending, tight otherwise
                if idx < len(dmsi_value) and not np.isnan(dmsi_value[idx]):
                    is_trending = dmsi_value[idx] >= dmsi_trend_level[idx]
                    active_mult = dmsi_wide_mult if is_trending else dmsi_tight_mult
                else:
                    active_mult = dmsi_wide_mult
                atr_val = fast_atr_array[idx]
                if not np.isnan(atr_val):
                    candidate = chandelier_extreme - active_mult * atr_val if is_bull \
                        else chandelier_extreme + active_mult * atr_val
                    if (is_bull and candidate > current_sl) or (not is_bull and candidate < current_sl):
                        current_sl = candidate
                        sl_moved = True

            elif trail_type == 8:
                # RATCHET_ATR_HYBRID -- deterministic ratchet floor first
                # (never gives back below the last locked milestone), then
                # STEPPED_ATR distance trail layered on top for tighter
                # management between milestones. current_sl takes whichever
                # candidate is more favorable each bar.
                while ratchet_next_step < n_ratchet_steps and \
                        max_favorable_r >= ratchet_trig[ratchet_next_step]:
                    lock_r = ratchet_lock[ratchet_next_step]
                    ratchet_candidate = entry + lock_r * risk if is_bull else entry - lock_r * risk
                    if (is_bull and ratchet_candidate > current_sl) or (not is_bull and ratchet_candidate < current_sl):
                        current_sl = ratchet_candidate
                        sl_moved = True
                    ratchet_next_step += 1

                if max_favorable_r >= step3_r:
                    active_mult = step3_mult
                elif max_favorable_r >= step2_r:
                    active_mult = step2_mult
                else:
                    active_mult = atr_mult
                atr_val = fast_atr_array[idx]
                if not np.isnan(atr_val):
                    atr_candidate = chandelier_extreme - active_mult * atr_val if is_bull \
                        else chandelier_extreme + active_mult * atr_val
                    if (is_bull and atr_candidate > current_sl) or (not is_bull and atr_candidate < current_sl):
                        current_sl = atr_candidate
                        sl_moved = True

        if not trail_armed and use_breakeven and not breakeven_armed:
            if is_bull and bar_high >= be_trigger_price:
                breakeven_armed = True
            elif not is_bull and bar_low <= be_trigger_price:
                breakeven_armed = True

        if breakeven_armed:
            hit_be = (bar_low <= be_stop) if is_bull else (bar_high >= be_stop)
            if hit_be:
                exit_price = be_stop
                r = (exit_price - entry) / risk if is_bull else (entry - exit_price) / risk
                return (1, r, max_favorable_r,
                        exit_price, tp, idx, True, trail_armed)

        if use_open_sl:
            sl_hit = (c[idx] <= current_sl) if is_bull else (c[idx] >= current_sl)
        else:
            sl_hit = (bar_low <= current_sl) if is_bull else (bar_high >= current_sl)

        if sl_hit:
            exit_price = c[idx] if use_open_sl else current_sl
            r = (exit_price - entry) / risk if is_bull else (entry - exit_price) / risk
            outcome = 3 if trail_armed else 2
            return (outcome, r, max_favorable_r,
                    exit_price, tp, idx, False, trail_armed)

        if has_tp:
            tp_hit = (bar_high >= tp) if is_bull else (bar_low <= tp)
            if tp_hit:
                return (4, rr, max_favorable_r,
                        tp, tp, idx, False, trail_armed)

    return (5, np.nan, max_favorable_r,
            np.nan, tp, -1, breakeven_armed, trail_armed)


@njit(cache=True)
def _max_fav_r_before_orig_sl(entry_idx, entry, is_bull, sl, n, h, l):
    """Independent scan (ignores BE/trail/TP/actual exit entirely): walks
    forward from entry and tracks the max favorable R reached, freezing
    the moment price first touches the ORIGINAL sl price (or running to
    the end of the data if it's never touched). This is intentionally
    decoupled from _simulate_trade_core's real exit logic -- current_sl
    only ever moves favorably relative to the original sl, so a touch of
    the original sl price and a current_sl-based exit are the same bar
    whenever the stop has moved; scanning independently is the only way
    to see how far price would have run had the original stop been left
    in place.
    """
    risk = abs(entry - sl)
    if risk <= 0.0:
        return 0.0

    max_fav = 0.0
    for idx in range(entry_idx + 1, n):
        bar_high = h[idx]
        bar_low = l[idx]

        fav_extreme = bar_high if is_bull else bar_low
        bar_fav_r = (fav_extreme - entry) / risk if is_bull else (entry - fav_extreme) / risk
        if bar_fav_r > max_fav:
            max_fav = bar_fav_r

        orig_sl_hit = (bar_low <= sl) if is_bull else (bar_high >= sl)
        if orig_sl_hit:
            break

    return max_fav


_OUTCOME_NAMES = {0: "invalid", 1: "breakeven", 2: "sl", 3: "trail", 4: "tp", 5: "open"}


def simulate_trade(hit, o, h, l, c, n, rr, use_breakeven, be_trigger, be_buffer,
                    atr_trigger, atr_period, atr_mult,
                    overlapping, use_open_sl, atr_array,
                    fast_atr_array, step2_r, step2_mult, step3_r, step3_mult,
                    trail_type, is_ph, is_pl, swing_buffer_mult,
                    ratchet_trig, ratchet_lock, n_ratchet_steps,
                    psar_af_start, psar_af_step, psar_af_max,
                    time_bars_full_tighten, time_mult_start, time_mult_end,
                    dmsi_value, dmsi_trend_level, dmsi_range_level,
                    dmsi_wide_mult, dmsi_tight_mult):
    entry_idx = hit["bar_index"]
    entry = float(c[entry_idx])
    is_bull = hit["direction"] == "bull"
    sl = float(hit["sl_extreme"])

    (outcome_code, r, max_fav_r,
     exit_price, tp, exit_idx, breakeven_hit, trail_active) = _simulate_trade_core(
        entry_idx, entry, is_bull, sl, n, rr,
        use_breakeven, be_trigger, be_buffer,
        atr_trigger, atr_mult,
        use_open_sl, h, l, c, atr_array,
        fast_atr_array, step2_r, step2_mult, step3_r, step3_mult,
        trail_type, is_ph, is_pl, swing_buffer_mult,
        ratchet_trig, ratchet_lock, n_ratchet_steps,
        psar_af_start, psar_af_step, psar_af_max,
        time_bars_full_tighten, time_mult_start, time_mult_end,
        dmsi_value, dmsi_trend_level, dmsi_range_level,
        dmsi_wide_mult, dmsi_tight_mult,
    )

    max_fav_r_before_sl = _max_fav_r_before_orig_sl(
        entry_idx, entry, is_bull, sl, n, h, l
    )

    if outcome_code == 0:  # zero-risk / invalid hit
        return {"outcome": "invalid", "r_multiple": 0.0, "max_favorable_r": 0.0,
                "max_favorable_r_before_sl": 0.0,
                "entry": entry, "sl": sl, "tp": None, "exit_price": None,
                "exit_index": None, "breakeven_hit": False, "trail_active": False}

    tp_out = None if (trail_type != 0 and outcome_code != 4) else float(tp)

    if outcome_code == 5:  # ran off the end of the data with no exit
        return {
            "outcome": "open", "r_multiple": None,
            "max_favorable_r": round(float(max_fav_r), 4),
            "max_favorable_r_before_sl": round(float(max_fav_r_before_sl), 4),
            "entry": entry, "sl": sl, "tp": tp_out,
            "exit_price": None, "exit_index": None,
            "breakeven_hit": bool(breakeven_hit), "trail_active": bool(trail_active),
        }

    return {
        "outcome": _OUTCOME_NAMES[outcome_code],
        "r_multiple": round(float(r), 4),
        "max_favorable_r": round(float(max_fav_r), 4),
        "max_favorable_r_before_sl": round(float(max_fav_r_before_sl), 4),
        "entry": entry, "sl": sl, "tp": tp_out,
        "exit_price": float(exit_price),
        "exit_index": int(exit_idx),
        "breakeven_hit": bool(breakeven_hit),
        "trail_active": bool(trail_active),
    }


# =====================================================================
#  CONSOLE TRADE / PERFORMANCE REPORTING
# =====================================================================
BOLD = "\033[1m"; GREEN = "\033[32m"; RED = "\033[31m"
YELLOW = "\033[33m"; CYAN = "\033[36m"; RESET = "\033[0m"


def box_line(line: str, width: int = 100):
    print(f"│ {line[:width-2]:<{width-2}} │")


TRAIL_TYPE_NAMES = {
    "OFF": 0,
    "STEPPED_ATR": 1,
    "CHANDELIER_ATR": 2,
    "STRUCTURE_SWING": 3,
    "R_RATCHET": 4,
    "PARABOLIC": 5,
    "TIME_BASED": 6,
    "DMSI_REGIME": 7,
    "RATCHET_ATR_HYBRID": 8,
}


def resolve_trail_type(name):
    """Maps cfg.TRAIL_TYPE (a string) to its integer code for
    _simulate_trade_core's trail_type dispatch. Raises ValueError on an
    unrecognized name so a config typo fails loudly."""
    if name not in TRAIL_TYPE_NAMES:
        raise ValueError(
            f"TRAIL_TYPE {name!r} not recognized. Valid values: "
            f"{sorted(TRAIL_TYPE_NAMES.keys())}"
        )
    return TRAIL_TYPE_NAMES[name]


def print_detailed_trades(hits, ctf_df, ctf_atr, ctf_fast_atr,
                           is_ph=None, is_pl=None,
                           dmsi_value=None, dmsi_trend_level=None, dmsi_range_level=None):
    n = len(ctf_df)
    o = ctf_df["open"].to_numpy(dtype=np.float64)
    h = ctf_df["high"].to_numpy(dtype=np.float64)
    l = ctf_df["low"].to_numpy(dtype=np.float64)
    c = ctf_df["close"].to_numpy(dtype=np.float64)

    trail_type = resolve_trail_type(getattr(cfg, "TRAIL_TYPE", "STEPPED_ATR"))

    # Structure-swing trail needs confirmed pivot arrays; fall back to
    # all-False arrays (never trails) if not supplied so the numba call
    # signature is always satisfiable regardless of trail_type.
    if is_ph is None:
        is_ph = np.zeros(n, dtype=bool)
    if is_pl is None:
        is_pl = np.zeros(n, dtype=bool)

    # DMSI-regime trail needs the DMSI arrays; fall back to NaN-filled
    # arrays (treated as "unknown regime" -> uses dmsi_wide_mult) if the
    # DMSI filter itself is disabled/unavailable upstream.
    if dmsi_value is None:
        dmsi_value = np.full(n, np.nan)
    if dmsi_trend_level is None:
        dmsi_trend_level = np.full(n, np.nan)
    if dmsi_range_level is None:
        dmsi_range_level = np.full(n, np.nan)

    ratchet_trig = np.asarray(getattr(cfg, "RATCHET_TRIGGER_R", []), dtype=np.float64)
    ratchet_lock = np.asarray(getattr(cfg, "RATCHET_LOCK_R", []), dtype=np.float64)
    if len(ratchet_trig) != len(ratchet_lock):
        raise ValueError(
            "RATCHET_TRIGGER_R and RATCHET_LOCK_R must be the same length "
            f"(got {len(ratchet_trig)} and {len(ratchet_lock)})"
        )
    n_ratchet_steps = len(ratchet_trig)

    overlapping = cfg.OVERLAPPING_TRADES
    blocked_until_idx = -1

    for hit in hits:
        if not overlapping and hit["bar_index"] < blocked_until_idx:
            hit["trade"] = {
                "outcome": "skipped_overlap", "r_multiple": None,
                "max_favorable_r": None, "max_favorable_r_before_sl": None,
                "entry": None, "sl": None, "tp": None,
                "exit_price": None, "exit_time": None,
                "breakeven_hit": False, "trail_active": False,
            }
            continue

        trade = simulate_trade(
            hit, o, h, l, c, n,
            rr=cfg.RR,
            use_breakeven=cfg.USE_BREAKEVEN,
            be_trigger=cfg.BREAKEVEN_TRIGGER,
            be_buffer=cfg.BREAKEVEN_BUFFER,
            atr_trigger=cfg.ATR_TRAIL_TRIGGER,
            atr_period=cfg.ATR_TRAIL_PERIOD,
            atr_mult=cfg.ATR_TRAIL_MULT,
            overlapping=overlapping,
            use_open_sl=cfg.USE_OPEN_SL,
            atr_array=ctf_atr,
            fast_atr_array=ctf_fast_atr,
            step2_r=cfg.ATR_TRAIL_STEP2_R,
            step2_mult=cfg.ATR_TRAIL_STEP2_MULT,
            step3_r=cfg.ATR_TRAIL_STEP3_R,
            step3_mult=cfg.ATR_TRAIL_STEP3_MULT,
            trail_type=trail_type,
            is_ph=is_ph,
            is_pl=is_pl,
            swing_buffer_mult=cfg.SWING_TRAIL_BUFFER_MULT,
            ratchet_trig=ratchet_trig,
            ratchet_lock=ratchet_lock,
            n_ratchet_steps=n_ratchet_steps,
            psar_af_start=cfg.PARABOLIC_AF_START,
            psar_af_step=cfg.PARABOLIC_AF_STEP,
            psar_af_max=cfg.PARABOLIC_AF_MAX,
            time_bars_full_tighten=cfg.TIME_TRAIL_BARS_FULL_TIGHTEN,
            time_mult_start=cfg.TIME_TRAIL_MULT_START,
            time_mult_end=cfg.TIME_TRAIL_MULT_END,
            dmsi_value=dmsi_value,
            dmsi_trend_level=dmsi_trend_level,
            dmsi_range_level=dmsi_range_level,
            dmsi_wide_mult=cfg.DMSI_TRAIL_WIDE_MULT,
            dmsi_tight_mult=cfg.DMSI_TRAIL_TIGHT_MULT,
        )

        if not overlapping:
            exit_idx = trade.get("exit_index")
            blocked_until_idx = (exit_idx + 1) if exit_idx is not None else n

        if trade.get("exit_index") is not None:
            trade["exit_time"] = ctf_df["time"].iloc[trade["exit_index"]].strftime("%Y-%m-%d %H:%M:%S")
            del trade["exit_index"]
        if cfg.USE_COMMISSION and trade.get("outcome") in ("tp", "sl", "breakeven", "trail"):
            trade["r_multiple_gross"] = trade["r_multiple"]
            trade["r_multiple"] = round(trade["r_multiple"] - cfg.COMMISSION_R, 4)
        hit["trade"] = trade

    W_SN = 5; W_TIME = 19; W_PATTERN = 15; W_DIR = 5; W_ENTRY = 13
    W_OUTCOME = 5; W_OUTCOME_R = 6; W_MFE = 8; W_MFEb4 = 12; W_CUM = 10

    show_outcome_r = getattr(cfg, "SHOW_OUTCOME_R", True)

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
        mfe_b4sl = t.get("max_favorable_r_before_sl")
        mfe_b4sl_str = f"{mfe_b4sl:+.2f}R" if mfe_b4sl is not None else "  --"
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
                   f"{mfe_b4sl_str:>{W_MFEb4}} {cum_colored} │")
        else:
            row = (f"│ {sn:<{W_SN}} {hit['time']:<{W_TIME}} {pat_colored} {dir_str} "
                   f"{entry_str} {out_colored} {mfe_str:>{W_MFE}} "
                   f"{mfe_b4sl_str:>{W_MFEb4}} {cum_colored} │")
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
    closed = [h["trade"] for h in hits if h["trade"]["outcome"] not in ("open", "invalid", "skipped_overlap")]
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
    skip_note = f"   (Skipped-overlap: {n_skipped})" if not cfg.OVERLAPPING_TRADES else ""
    print(f"  Total closed trades: {n_closed}   (Open: {n_open}){skip_note}")
    print(f"  Wins: {n_wins}   Losses: {n_losses}")
    if getattr(cfg, "TRAIL_TYPE", "STEPPED_ATR") != "OFF":
        print(f"  Trail exits: {n_trail}   (Trail wins: {trail_wins}   Trail losses: {trail_losses})")
    print(f"  Win rate: {win_rate:.2f}%")
    print(f"  Total R: {total_r:+.2f}   Expectancy: {expectancy:+.3f}R")
    print("-"*70)

    # Monthly stats
    months = {}
    for hit in hits:
        t = hit["trade"]
        if t["outcome"] in ("open", "invalid", "skipped_overlap"): continue
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
        if t["outcome"] in ("open", "invalid", "skipped_overlap"): continue
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
        if t["outcome"] in ("open", "invalid", "skipped_overlap"): continue
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
        if t["outcome"] in ("open", "invalid", "skipped_overlap"): continue
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
        if t["outcome"] in ("open", "invalid", "skipped_overlap"): continue
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
    if hits:
        days_count = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days + 1
    else:
        days_count = 0
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

    if getattr(cfg, "SHOW_DAILY_BREAKDOWN", False):
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


# =====================================================================
# =====================================================================
#  SCANNER PIPELINE
#  (CSV loading, HTF resampling, running the engine over the requested
#  date range, console report, JSON report)
# =====================================================================
# =====================================================================

TF_TO_PANDAS = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H2": "2h", "H3": "3h", "H4": "4h", "D1": "1D",
}

# Native bar spacing (minutes) each timeframe label corresponds to, used to
# auto-detect a CSV's actual resolution from its timestamp spacing.
TF_TO_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H2": 120, "H3": 180, "H4": 240, "D1": 1440,
}


def htf_close_times_from_open(htf_times, tf_name):
    """Returns each HTF bar's CLOSE time (open time + declared bar
    duration) given its open times and a TF_TO_MINUTES key. Used to fix
    look-ahead bias in HTF-to-CTF as-of joins: a CTF bar must only ever
    see an HTF bar's data (trend, zones) after that HTF bar has actually
    finished forming, not from the instant it opens. Duration comes from
    the declared timeframe rather than inferred bar-to-bar spacing, so it
    stays correct even for the final bar or across data gaps."""
    minutes = TF_TO_MINUTES.get(tf_name)
    if minutes is None:
        raise ValueError(f"Unknown timeframe '{tf_name}' -- must be one of {sorted(TF_TO_MINUTES)}")
    return htf_times + np.timedelta64(minutes, "m")


def detect_native_timeframe(df):
    """Infers a CSV's native bar timeframe from the median spacing between
    consecutive timestamps (median rather than the first gap, so it isn't
    thrown off by a handful of missing/duplicate bars or a gap at the very
    start). Returns a TF_TO_MINUTES key (e.g. 'M1', 'M5') and the raw
    median-gap-in-minutes value. Raises if the spacing doesn't match any
    known timeframe."""
    if len(df) < 2:
        raise ValueError("CSV has fewer than 2 rows -- can't detect timeframe")

    deltas = df["time"].diff().dropna()
    median_minutes = deltas.median().total_seconds() / 60.0

    for tf, minutes in TF_TO_MINUTES.items():
        if abs(median_minutes - minutes) < 1e-6:
            return tf, median_minutes

    raise ValueError(
        f"CSV's native bar spacing (median {median_minutes:.3f} min) doesn't "
        f"match any known timeframe {sorted(TF_TO_MINUTES.items(), key=lambda kv: kv[1])}. "
        f"Check the CSV for gaps, duplicate timestamps, or an unsupported resolution."
    )


# =====================================================================
#  DATA LOADING
# =====================================================================
def load_csv(path, date_format=None):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "time" in df.columns:
        time_col = "time"
    elif "date" in df.columns:
        time_col = "date"
    else:
        raise ValueError("CSV must have a 'time' or 'date' column")

    df["time"] = pd.to_datetime(df[time_col], format=date_format)
    df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)

    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    keep = ["time"] + required
    if "tick_volume" in df.columns:
        keep.append("tick_volume")
    return df[keep]


def resample_ohlc(df, pandas_freq):
    """Resamples closed-bar OHLC data to a higher timeframe using standard
    OHLC aggregation (bucket labeled by its start time, matching MT5's
    left-closed HTF bar convention)."""
    d = df.set_index("time")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "tick_volume" in d.columns:
        agg["tick_volume"] = "sum"
    out = d.resample(pandas_freq, label="left", closed="left").agg(agg).dropna()
    out = out.reset_index()
    return out


# =====================================================================
#  MAIN PIPELINE
# =====================================================================
def run_scan():
    print("=" * 70)
    print("TrendStructure + PD Zones -- Offline Scanner")
    print("=" * 70)

    # --- Load full CSV (we need history before SCAN_START for indicator warm-up) ---
    print(f"Loading CSV: {cfg.CSV_PATH}")
    full_df = load_csv(cfg.CSV_PATH, cfg.CSV_DATE_FORMAT)
    print(f"  {len(full_df)} bars loaded, "
          f"{full_df['time'].iloc[0]} -> {full_df['time'].iloc[-1]}")

    # --- Determine HTF mapping ---
    ctf_name = cfg.CHART_TIMEFRAME
    htf_name = map_timeframe(ctf_name) if cfg.UseHTFMapping else ctf_name
    pdz_htf_name = map_timeframe(ctf_name) if cfg.InpUseHTFZones else ctf_name

    print(f"Chart TF: {ctf_name}  |  Trend HTF: {htf_name}  |  PD-Zones HTF: {pdz_htf_name}")
    print(f"Trend Engine: {cfg.TrendEngine}")

    # --- Detect CSV's native resolution and resample to CHART_TIMEFRAME if needed ---
    native_tf, native_minutes = detect_native_timeframe(full_df)
    target_minutes = TF_TO_MINUTES.get(ctf_name)
    if target_minutes is None:
        raise ValueError(f"Unknown CHART_TIMEFRAME '{ctf_name}' -- must be one of {sorted(TF_TO_MINUTES)}")

    if native_tf == ctf_name:
        print(f"CSV native timeframe: {native_tf} (matches CHART_TIMEFRAME, no resample needed)")
        ctf_df = full_df.reset_index(drop=True)
    elif target_minutes > native_minutes:
        print(f"CSV native timeframe: {native_tf} -- resampling up to CHART_TIMEFRAME {ctf_name}")
        ctf_df = resample_ohlc(full_df, TF_TO_PANDAS[ctf_name])
    else:
        raise ValueError(
            f"CSV native timeframe is {native_tf} ({native_minutes:.0f} min/bar), which is coarser "
            f"than CHART_TIMEFRAME {ctf_name} ({target_minutes:.0f} min/bar). Can't fabricate finer "
            f"bars from coarser data -- lower CHART_TIMEFRAME to {native_tf} or below, or use a "
            f"finer-resolution CSV."
        )
    print(f"  CTF({ctf_name}) bars: {len(ctf_df)}")
    ctf_times = ctf_df["time"].values
    ctf_open = ctf_df["open"].to_numpy(dtype=float)
    ctf_high = ctf_df["high"].to_numpy(dtype=float)
    ctf_low = ctf_df["low"].to_numpy(dtype=float)
    ctf_close = ctf_df["close"].to_numpy(dtype=float)
    n_ctf = len(ctf_df)

    # --- Build HTF (trend engine) series via resample ---
    if htf_name == ctf_name:
        htf_df = ctf_df
    else:
        htf_df = resample_ohlc(full_df, TF_TO_PANDAS[htf_name])
    htf_times = htf_df["time"].values
    htf_high = htf_df["high"].to_numpy(dtype=float)
    htf_low = htf_df["low"].to_numpy(dtype=float)
    htf_close = htf_df["close"].to_numpy(dtype=float)
    n_htf = len(htf_df)
    print(f"  HTF({htf_name}) bars: {n_htf}")

    # --- Build PDZ HTF series (may be a different mapping target if InpUseHTFZones toggled independently) ---
    if pdz_htf_name == ctf_name:
        pdz_htf_df = ctf_df
    elif pdz_htf_name == htf_name:
        pdz_htf_df = htf_df
    else:
        pdz_htf_df = resample_ohlc(full_df, TF_TO_PANDAS[pdz_htf_name])
    pdz_htf_times = pdz_htf_df["time"].values
    pdz_htf_high = pdz_htf_df["high"].to_numpy(dtype=float)
    pdz_htf_low = pdz_htf_df["low"].to_numpy(dtype=float)
    pdz_htf_close = pdz_htf_df["close"].to_numpy(dtype=float)

    # =================================================================
    # 1) TREND ENGINE on HTF
    # =================================================================
    print("\nComputing trend structure...")
    htf_trend = compute_trend(
        htf_high, htf_low, htf_close, n_htf,
        cfg.PivotLength, cfg.RequiredPivotPairs, cfg.HoldLastTrendOnUndefined,
        cfg.TrendEngine,
    )

    trend_age, price_pos, price_pos_valid = compute_trend_age(
        htf_high, htf_low, htf_close, htf_trend, n_htf, cfg.PivotLength
    )

    pivot_label_events = []
    if cfg.ShowPivotLabels:
        pivot_label_events = compute_pivot_labels(htf_high, htf_low, htf_times, n_htf, cfg.PivotLength)

    # CTF trend (only needed if FilterHTFAlignment is on)
    ctf_trend = None
    if cfg.FilterHTFAlignment:
        ctf_trend = compute_trend(
            ctf_high, ctf_low, ctf_close, n_ctf,
            cfg.PivotLength, cfg.RequiredPivotPairs, cfg.HoldLastTrendOnUndefined,
            cfg.TrendEngine,
        )

    # =================================================================
    # 2) MAP HTF TREND -> EACH CTF BAR  (mirrors the pre-paint loop)
    # =================================================================
    print("Mapping HTF trend onto chart-timeframe bars...")
    # Use each HTF bar's CLOSE time (not open time) as the join cutoff, so a
    # CTF bar only ever sees its HTF parent's trend after that HTF bar has
    # actually finished forming -- matches MT5's live-bar exclusion
    # (CopyRates shift=1) and avoids leaking a still-forming HTF bar's
    # eventual trend/high/low back onto earlier CTF bars.
    htf_close_times = htf_close_times_from_open(htf_times, htf_name)
    last_closed_htf_idx = n_htf - 1
    bar_trend = np.zeros(n_ctf, dtype=int)
    bar_trend_age = np.zeros(n_ctf, dtype=int)
    bar_price_pos = np.full(n_ctf, -1.0)

    htf_idx = 0
    for i in range(n_ctf):
        t = ctf_times[i]
        while htf_idx + 1 <= last_closed_htf_idx and htf_close_times[htf_idx + 1] <= t:
            htf_idx += 1
        while htf_idx > 0 and htf_close_times[htf_idx] > t:
            htf_idx -= 1
        use_idx = min(htf_idx, last_closed_htf_idx)
        if use_idx >= 0 and htf_close_times[use_idx] <= t:
            bar_trend[i] = htf_trend[use_idx]
            bar_trend_age[i] = trend_age[use_idx]
            bar_price_pos[i] = price_pos[use_idx] if price_pos_valid[use_idx] else -1.0
        else:
            bar_trend[i] = 0

    color_buf = np.where(bar_trend == 1, 1, np.where(bar_trend == -1, 2, 0))

    if cfg.FilterHTFAlignment and ctf_trend is not None:
        last_closed_ctf_idx = n_ctf - 1
        ctf_idx = 0
        for i in range(n_ctf):
            t = ctf_times[i]
            while ctf_idx + 1 <= last_closed_ctf_idx and ctf_times[ctf_idx + 1] <= t:
                ctf_idx += 1
            use_idx = min(ctf_idx, last_closed_ctf_idx)
            ctf_trend_here = ctf_trend[use_idx] if ctf_times[use_idx] <= t else 0
            both_defined = (bar_trend[i] != 0 and ctf_trend_here != 0)
            if both_defined and bar_trend[i] != ctf_trend_here:
                color_buf[i] = 4  # HTF disagreement -> neutral/gray

    # =================================================================
    # 3) PD ZONES
    # =================================================================
    print("Computing PD zones...")
    if cfg.InpUseHTFZones:
        pdz_zones_src = compute_zones(
            pdz_htf_high, pdz_htf_low, pdz_htf_close, len(pdz_htf_df),
            cfg.InpLookbackPeriod, cfg.InpUseATR, cfg.InpATRPeriod, cfg.InpATRMultiplier,
            cfg.InpUseManualZones, cfg.InpDiscountInnerOffset, cfg.InpDiscountOuterOffset,
            cfg.InpPremiumInnerOffset, cfg.InpPremiumOuterOffset,
            cfg.POINT_SIZE, cfg.InpBufferPoints,
        )
        # Close time, not open time -- see htf_close_times_from_open docstring;
        # otherwise a CTF bar could see its own still-forming PDZ-HTF parent's
        # zone (computed from that parent's full, not-yet-happened range).
        pdz_htf_close_times = htf_close_times_from_open(pdz_htf_times, pdz_htf_name)
        eq, p_top, p_bot, d_top, d_bot = map_ctf_zones_from_htf(
            ctf_times, pdz_htf_close_times, pdz_zones_src, n_ctf
        )
    else:
        zones = compute_zones(
            ctf_high, ctf_low, ctf_close, n_ctf,
            cfg.InpLookbackPeriod, cfg.InpUseATR, cfg.InpATRPeriod, cfg.InpATRMultiplier,
            cfg.InpUseManualZones, cfg.InpDiscountInnerOffset, cfg.InpDiscountOuterOffset,
            cfg.InpPremiumInnerOffset, cfg.InpPremiumOuterOffset,
            cfg.POINT_SIZE, cfg.InpBufferPoints,
        )
        eq = np.where(zones["valid"], zones["eq"], np.nan)
        p_top = np.where(zones["valid"], zones["premium_top"], np.nan)
        p_bot = np.where(zones["valid"], zones["premium_bot"], np.nan)
        d_top = np.where(zones["valid"], zones["discount_top"], np.nan)
        d_bot = np.where(zones["valid"], zones["discount_bot"], np.nan)

    # =================================================================
    # 4) ZONE-TOUCH FILTER (downgrades green/red -> neutral if no zone interaction)
    # =================================================================
    if cfg.InpRequireZoneTouch:
        overlap_threshold = max(cfg.InpZoneOverlapPercent, 0.0) / 100.0
        for i in range(n_ctf):
            if color_buf[i] not in (1, 2):
                continue
            if color_buf[i] == 1:
                zl, zh = d_bot[i], d_top[i]
            else:
                zl, zh = p_bot[i], p_top[i]
            if not zone_touch_allows(ctf_close[i], ctf_low[i], ctf_high[i], zl, zh, overlap_threshold):
                color_buf[i] = 0

    # =================================================================
    # 4b) DMSI REGIME FILTER (stacks on top of trend-alignment + zone-touch;
    #     downgrades whatever those two already let through if the market
    #     isn't in a DMSI "Trending" regime)
    # =================================================================
    dmsi_value = dmsi_trend_level = dmsi_range_level = None
    if getattr(cfg, "InpUseDMSIFilter", False):
        print("Computing DMSI regime filter...")
        dmsi_value, dmsi_trend_level, dmsi_range_level = compute_dmsi(
            ctf_high, ctf_low, ctf_close, n_ctf,
            cfg.InpDMSILookback, cfg.InpDMSIPercentilePeriod,
            cfg.InpDMSITrendPercentile, cfg.InpDMSIRangePercentile,
            cfg.InpDMSIFastSmooth, cfg.InpDMSISlowSmooth,
        )
        dmsi_min_bars = cfg.InpDMSILookback * 2 + 10
        for i in range(n_ctf):
            if color_buf[i] not in (1, 2):
                continue
            if i < dmsi_min_bars:
                color_buf[i] = 0  # not enough history for a regime read yet
                continue
            if dmsi_regime_at(dmsi_value, dmsi_trend_level, dmsi_range_level, i) != 0:
                color_buf[i] = 0

    # =================================================================
    # 4c) WARM-UP GATE
    #     Hard-stops if trend/DMSI/zones haven't had enough real history
    #     before SCAN_START to be past cold-start. This does not change
    #     any indicator math -- it only refuses to run the scan itself.
    # =================================================================
    _scan_start_ts = pd.Timestamp(cfg.SCAN_START) if cfg.SCAN_START else pd.Timestamp(ctf_times[0])

    # How many HTF bars exist strictly before scan_start?
    _htf_before_start = int(np.searchsorted(htf_times, np.datetime64(_scan_start_ts), side="left"))

    _trend_min_htf_bars = cfg.PivotLength * 2 + 5
    _warmup_floor = getattr(cfg, "WARMUP_HTF_BARS", None)
    _required_htf_bars = _trend_min_htf_bars if not _warmup_floor else max(_trend_min_htf_bars, _warmup_floor)

    if _htf_before_start < _required_htf_bars:
        _deficit = _required_htf_bars - _htf_before_start
        _first_htf_time = fmt_time(htf_times[0]) if n_htf > 0 else "N/A"
        raise WarmupError(
            f"Not enough HTF({htf_name}) history before SCAN_START for the trend "
            f"engine to warm up.\n"
            f"  HTF bars available before {_scan_start_ts}: {_htf_before_start}\n"
            f"  HTF bars required (PivotLength*2+5{' , floor=' + str(_warmup_floor) if _warmup_floor else ''}): {_required_htf_bars}\n"
            f"  Deficit: {_deficit} bars\n"
            f"  HTF series starts at: {_first_htf_time}\n"
            f"  Either move SCAN_START later, extend the CSV's history earlier, "
            f"or lower PivotLength / WARMUP_HTF_BARS."
        )

    if getattr(cfg, "InpUseDMSIFilter", False):
        _dmsi_min_bars = cfg.InpDMSILookback * 2 + 10
        _ctf_before_start = int(np.searchsorted(ctf_times, np.datetime64(_scan_start_ts), side="left"))
        if _ctf_before_start < _dmsi_min_bars:
            _deficit = _dmsi_min_bars - _ctf_before_start
            _first_ctf_time = fmt_time(ctf_times[0]) if n_ctf > 0 else "N/A"
            raise WarmupError(
                f"Not enough CTF({ctf_name}) history before SCAN_START for the "
                f"DMSI regime filter to warm up.\n"
                f"  CTF bars available before {_scan_start_ts}: {_ctf_before_start}\n"
                f"  CTF bars required (InpDMSILookback*2+10): {_dmsi_min_bars}\n"
                f"  Deficit: {_deficit} bars\n"
                f"  CTF series starts at: {_first_ctf_time}\n"
                f"  Either move SCAN_START later, extend the CSV's history earlier, "
                f"or lower InpDMSILookback / disable InpUseDMSIFilter."
            )

    # =================================================================
    # 5) SIGNAL ARROWS (buy/sell), only if enabled
    # =================================================================
    buy_signals = []
    sell_signals = []
    if cfg.InpShowSignals:
        print("Scanning signal arrows...")
        for i in range(n_ctf - 1):
            if math.isnan(eq[i]):
                continue
            if cfg.InpUseManualZones:
                buf = cfg.InpDiscountInnerOffset
            else:
                buf = (p_bot[i] - eq[i]) if not math.isnan(p_bot[i]) else 0
            if buf == 0:
                continue

            # BUY
            if ctf_close[i] < eq[i] - buf:
                confluence = False
                recent_low = ctf_low[i]
                for j in range(1, cfg.InpConfluenceLookback + 1):
                    if i - j < 0:
                        break
                    if ctf_low[i - j] < recent_low:
                        recent_low = ctf_low[i - j]
                if ctf_low[i] <= recent_low + cfg.POINT_SIZE and ctf_close[i] > recent_low:
                    confluence = True
                if not confluence:
                    for j in range(2, cfg.InpConfluenceLookback + 1):
                        if i - j < 0:
                            break
                        if ctf_close[i - j] >= ctf_open[i - j]:
                            continue
                        impulse = 0
                        for k in range(1, 4):
                            if i - j + k > i:
                                break
                            v = ctf_close[i - j + k] - ctf_close[i - j]
                            if v > impulse:
                                impulse = v
                        if impulse >= cfg.InpMinImpulsePoints * cfg.POINT_SIZE:
                            confluence = True
                            break
                is_bull_rev = (ctf_close[i] > ctf_open[i]) and \
                              ((ctf_close[i] - ctf_open[i]) / max(ctf_high[i] - ctf_low[i], 1e-12) > 0.4)
                if confluence and is_bull_rev:
                    buy_signals.append({"bar_time": fmt_time(ctf_times[i]), "price": float(ctf_low[i] - buf * 0.3)})

            # SELL
            if ctf_close[i] > eq[i] + buf:
                confluence = False
                recent_high = ctf_high[i]
                for j in range(1, cfg.InpConfluenceLookback + 1):
                    if i - j < 0:
                        break
                    if ctf_high[i - j] > recent_high:
                        recent_high = ctf_high[i - j]
                if ctf_high[i] >= recent_high - cfg.POINT_SIZE and ctf_close[i] < recent_high:
                    confluence = True
                if not confluence:
                    for j in range(2, cfg.InpConfluenceLookback + 1):
                        if i - j < 0:
                            break
                        if ctf_close[i - j] <= ctf_open[i - j]:
                            continue
                        impulse = 0
                        for k in range(1, 4):
                            if i - j + k > i:
                                break
                            v = ctf_close[i - j] - ctf_close[i - j + k]
                            if v > impulse:
                                impulse = v
                        if impulse >= cfg.InpMinImpulsePoints * cfg.POINT_SIZE:
                            confluence = True
                            break
                is_bear_rev = (ctf_close[i] < ctf_open[i]) and \
                              ((ctf_open[i] - ctf_close[i]) / max(ctf_high[i] - ctf_low[i], 1e-12) > 0.4)
                if confluence and is_bear_rev:
                    sell_signals.append({"bar_time": fmt_time(ctf_times[i]), "price": float(ctf_high[i] + buf * 0.3)})

    # =================================================================
    # 6) CANDLE PATTERNS
    # =================================================================
    pattern_events = []
    if cfg.ShowPatternLabels:
        enabled_patterns = resolve_enabled_patterns(getattr(cfg, "ENABLED_PATTERNS", None))
        excluded_patterns = resolve_excluded_patterns(getattr(cfg, "EXCLUDE_PATTERNS", None))
        enabled_patterns = apply_pattern_exclusions(enabled_patterns, excluded_patterns)
        if enabled_patterns is None:
            print("Scanning candle patterns... (all patterns)")
        else:
            scanned_names = [PATTERN_NAMES[p] for p in sorted(enabled_patterns)]
            print(f"Scanning candle patterns... (filtered to: {scanned_names})")
        pattern_events = compute_pattern_events(
            ctf_times, ctf_open, ctf_high, ctf_low, ctf_close, color_buf, n_ctf,
            cfg.PatternDojiBodyRatio, cfg.PatternSpinningTopBodyRatio, cfg.PatternLongBodyRatio,
            cfg.PatternRequireHtfAgreement, htf_close_times, htf_trend,
            enabled_pattern_indices=enabled_patterns,
        )

    # =================================================================
    # 7) TREND FLIP EVENTS (for alerts / console report)
    # =================================================================
    trend_flip_events = []
    prev_trend = None
    for i in range(n_htf):
        t = htf_trend[i]
        if prev_trend is not None and t != prev_trend and t != 0:
            trend_flip_events.append({
                "bar_time": fmt_time(htf_times[i]),
                "direction": "BULLISH" if t == 1 else "BEARISH",
            })
        prev_trend = t

    # =================================================================
    # 8) FILTER EVERYTHING TO THE REQUESTED SCAN WINDOW
    # =================================================================
    scan_start = pd.Timestamp(cfg.SCAN_START) if cfg.SCAN_START else ctf_times[0]
    scan_end = pd.Timestamp(cfg.SCAN_END) if cfg.SCAN_END else ctf_times[-1]

    def in_range(ts_str):
        ts = pd.Timestamp(ts_str)
        return scan_start <= ts <= scan_end

    mask = (ctf_times >= np.datetime64(scan_start)) & (ctf_times <= np.datetime64(scan_end))
    scan_idx = np.where(mask)[0]

    trend_flip_events = [e for e in trend_flip_events if in_range(e["bar_time"])]
    pivot_label_events = [e for e in pivot_label_events if in_range(e["pivot_bar_time"])]
    pattern_events = [e for e in pattern_events if in_range(e["bar_time"])]
    buy_signals = [e for e in buy_signals if in_range(e["bar_time"])]
    sell_signals = [e for e in sell_signals if in_range(e["bar_time"])]

    # Per-bar snapshot table for the scan window
    bars_report = []
    dmsi_min_bars = cfg.InpDMSILookback * 2 + 10 if getattr(cfg, "InpUseDMSIFilter", False) else None
    _show_bars_debug = getattr(cfg, "SHOW_BARS_DEBUG", True)
    if _show_bars_debug:
        for i in scan_idx:
            color_name = {0: "neutral", 1: "bullish", 2: "bearish", 4: "htf_disagree"}[int(color_buf[i])]
            dmsi_regime_name = None
            if dmsi_value is not None and i >= dmsi_min_bars:
                dmsi_regime_name = {0: "trending", 1: "transition", 2: "ranging"}[
                    dmsi_regime_at(dmsi_value, dmsi_trend_level, dmsi_range_level, i)
                ]
            bars_report.append({
                "time": fmt_time(ctf_times[i]),
                "open": float(ctf_open[i]), "high": float(ctf_high[i]),
                "low": float(ctf_low[i]), "close": float(ctf_close[i]),
                "trend": int(bar_trend[i]),
                "color": color_name,
                "trend_age": int(bar_trend_age[i]),
                "price_position": float(bar_price_pos[i]) if bar_price_pos[i] >= 0 else None,
                "eq": float(eq[i]) if not math.isnan(eq[i]) else None,
                "premium_top": float(p_top[i]) if not math.isnan(p_top[i]) else None,
                "premium_bot": float(p_bot[i]) if not math.isnan(p_bot[i]) else None,
                "discount_top": float(d_top[i]) if not math.isnan(d_top[i]) else None,
                "discount_bot": float(d_bot[i]) if not math.isnan(d_bot[i]) else None,
                "dmsi_regime": dmsi_regime_name,
            })

    # =================================================================
    # 9) CONSOLE REPORT
    # =================================================================
    print("\n" + "=" * 70)
    print(f"SCAN WINDOW: {scan_start} -> {scan_end}")
    print(f"Bars in window: {len(scan_idx)}")
    print("=" * 70)

    if cfg.ShowPivotLabels:
        print(f"\n--- PIVOT LABELS ({len(pivot_label_events)}) ---")
        for e in pivot_label_events[:50]:
            print(f"  {e['confirmed_bar_time']}  {e['type']}  @ {e['price']:.5f}  (pivot bar {e['pivot_bar_time']})")
        if len(pivot_label_events) > 50:
            print(f"  ... {len(pivot_label_events) - 50} more (see JSON)")

    if cfg.ShowPatternLabels:
        # ---------------------------------------------------------------
        # TRADE SIMULATION: each pattern event within the scan window is
        # a trade hit (bar_index / direction / sl_extreme already attached
        # in compute_pattern_events). Simulated forward-walk over the full
        # CTF series (not just the scan window) so trades can exit after
        # the scan window ends.
        # ---------------------------------------------------------------
        hits = pattern_events
        if hits:
            print("\nSimulating trades...")
            ctf_atr = precompute_atr(ctf_df, cfg.ATR_TRAIL_PERIOD)
            ctf_fast_atr = precompute_atr(ctf_df, cfg.ATR_TRAIL_FAST_PERIOD)
            # STRUCTURE_SWING trail needs confirmed pivot arrays over the
            # same CTF series used for simulation; reuses the same
            # precompute_pivots() the trend engine itself is built on.
            trail_is_ph, trail_is_pl = precompute_pivots(
                ctf_high, ctf_low, cfg.PivotLength, n_ctf
            )
            net_r = print_detailed_trades(
                hits, ctf_df, ctf_atr, ctf_fast_atr,
                is_ph=trail_is_ph, is_pl=trail_is_pl,
                dmsi_value=dmsi_value, dmsi_trend_level=dmsi_trend_level,
                dmsi_range_level=dmsi_range_level,
            )
            print_performance_stats(hits, net_r)

    if cfg.InpShowSignals:
        print(f"\n--- BUY SIGNALS ({len(buy_signals)}) ---")
        for e in buy_signals[:30]:
            print(f"  {e['bar_time']}  @ {e['price']:.5f}")
        print(f"\n--- SELL SIGNALS ({len(sell_signals)}) ---")
        for e in sell_signals[:30]:
            print(f"  {e['bar_time']}  @ {e['price']:.5f}")

    _scan_colors = color_buf[scan_idx]
    n_bull = int(np.count_nonzero(_scan_colors == 1))
    n_bear = int(np.count_nonzero(_scan_colors == 2))
    n_neutral = int(np.count_nonzero(_scan_colors == 0))
    n_disagree = int(np.count_nonzero(_scan_colors == 4))
    print(f"\n--- CANDLE COLOR SUMMARY ---")
    print(f"  bullish: {n_bull}   bearish: {n_bear}   neutral: {n_neutral}   htf_disagree: {n_disagree}")
    if getattr(cfg, "InpUseDMSIFilter", False):
        print(f"  (DMSI regime filter: ON -- only Trending regime bars retain color)")

    # =================================================================
    # 10) SAVE JSON REPORT
    # =================================================================
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    if cfg.RESULTS_JSON_NAME:
        out_name = cfg.RESULTS_JSON_NAME
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"scan.json"
    out_path = os.path.join(cfg.RESULTS_DIR, out_name)

    report = {
        "config": {
            "csv_path": cfg.CSV_PATH,
            "chart_timeframe": ctf_name,
            "trend_htf": htf_name,
            "pdz_htf": pdz_htf_name,
            "trend_engine": cfg.TrendEngine,
            "pivot_length": cfg.PivotLength,
            "required_pivot_pairs": cfg.RequiredPivotPairs,
            "dmsi_filter_enabled": bool(getattr(cfg, "InpUseDMSIFilter", False)),
            "scan_start": str(scan_start),
            "scan_end": str(scan_end),
        },
        "summary": {
            "bars_in_window": len(scan_idx),
            "bullish_bars": n_bull,
            "bearish_bars": n_bear,
            "neutral_bars": n_neutral,
            "htf_disagree_bars": n_disagree,
            "trend_flip_count": len(trend_flip_events),
            "pattern_count": len(pattern_events),
            "buy_signal_count": len(buy_signals),
            "sell_signal_count": len(sell_signals),
        },
        "trend_flips": trend_flip_events,
        "pivot_labels": pivot_label_events,
        "candle_patterns": pattern_events,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "bars": bars_report,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_scan()