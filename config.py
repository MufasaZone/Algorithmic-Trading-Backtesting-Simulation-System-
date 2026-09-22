"""
config.py -- TrendStructure + PDZones filter + MQL5-exact Japanese pattern scanner + Trade Simulator
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
SYMBOL = "Volatility 30 (1s) Index.0_M1"
# SYMBOL = "Volatility 10 (1s) Index.0_M1"
# SYMBOL = "Volatility 5 (1s) Index.0_M1"



# SYMBOL = "Volatility 30 Index.0_M1"
# SYMBOL = "Volatility 5 Index.0_M1"

# SYMBOL = "Jump 25 Index.0_M1"
# SYMBOL = "EURUSD_M5"


_CONFIG_DIR = Path(__file__).resolve().parent
CSV_PATH = str(_CONFIG_DIR / ".." / "CSV_deriv_zero_spread" / "csv" / f"{SYMBOL}.csv")

CTF_TIMEFRAME = "5min"
# CTF_TIMEFRAME = "7min"

# ---------------------------------------------------------------------------
# Multi-timeframe analysis
# ---------------------------------------------------------------------------
# If non-empty, the scanner runs the full scan + simulation once per entry
# here (each writes its own result/{SYMBOL}_{timeframe}.json) and prints a
# comparison table at the end. Leave empty (or None) to run once using
# CTF_TIMEFRAME above only, same as before.
# TIMEFRAMES = []
TIMEFRAMES = [
    "3min",
    "4min", 
    "5min", 
    "6min", 
    "10min", 
    "15min", 
    "30min"
    ]


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------
# START_DATE = "2025-12-23"
# END_DATE   = "2025-12-23 23:59:00"

# START_DATE = "2025-12-01"
# END_DATE   = "2025-12-28 23:59:00"

# START_DATE = "2025-03-02"
# END_DATE   = "2025-03-06 23:59:00"

# START_DATE = "2025-01-01"
# END_DATE   = "2026-12-29 23:59:00"

START_DATE = "2025-01-01"
END_DATE   = "2025-12-29 23:59:00"




# ---------------------------------------------------------------------------
# Trade parameters
# ---------------------------------------------------------------------------
# RR = 6
RR = 9
USE_BREAKEVEN = True
# USE_BREAKEVEN = False
# BREAKEVEN_TRIGGER = 2.8
BREAKEVEN_TRIGGER = 3
BREAKEVEN_BUFFER = 0.1
# SL_BUFFER_PCT = 0.0
SL_BUFFER_PCT = 0.03
# SL_BUFFER_PCT = 0.005

# ---------------------------------------------------------------------------
# Trailing Stop
# ---------------------------------------------------------------------------
# Pick ONE trailing mode. The SL only ever tightens (moves in the trade's
# favor) — it will never move backwards regardless of mode.
#   "none"       - no trailing, fixed SL only (+ optional single-step breakeven above)
#   "staircase"  - move SL to lock_r once price reaches trigger_r (multi-step breakeven)
#   "atr"        - trail behind price by a multiple of ATR (volatility-adaptive)
#   "structure"  - trail to the most recently confirmed swing low/high (price-structure based)
TRAIL_MODE = "none"

# If True, trailing only starts once the existing USE_BREAKEVEN trigger has
# armed (i.e. breakeven-then-trail). If False, trailing runs independently
# of USE_BREAKEVEN from trade entry onward.
TRAIL_START_AFTER_BREAKEVEN = True

# --- staircase mode ---
# List of (trigger_r, lock_r) pairs: once max favorable R reaches trigger_r,
# SL is moved to lock_r (expressed in R from entry). Steps are evaluated
# independently each bar and the most favorable applicable lock wins, so
# order doesn't matter, but ascending trigger_r reads best.
TRAIL_STAIRCASE_STEPS = [
    (2.0, 1.0),
    (4.0, 2.0),
    (6.0, 4.0),
    (10, 6.0),
    (15.0, 10.0),
    (19.0, 18.0),
]

# --- atr mode ---
TRAIL_ATR_PERIOD = 14
TRAIL_ATR_MULTIPLIER = 2.0
TRAIL_ATR_MIN_ACTIVATE_R = 1.0   # don't start ATR-trailing until this many R in favor (0 = immediately)

# --- structure mode ---
# Reuses PIVOT_LENGTH from the Trend Structure settings below to confirm swings.
TRAIL_STRUCTURE_BUFFER_PCT = 0.05   # extra room beyond the swing level, as % of entry price

# ---------------------------------------------------------------------------
# SMA Trend Strength Oscillator Filter
# ---------------------------------------------------------------------------
# VOl 30 (1s)
USE_SMA_TREND_FILTER = True            # Enable/disable the filter
SMA_FAST_PERIOD       = 20
SMA_SLOW_PERIOD       = 150
SMA_SIGNAL_PERIOD     = 3
SMA_RANGE_FILTER_PERIOD = 100
SMA_RANGE_FILTER_MULTIPLIER = 1.5
# SMA_RANGE_FILTER_MULTIPLIER = 0.5
SMA_REQUIRE_ALIGNMENT = True           # If True, require pattern direction to match oscillator colour


# USE_SMA_TREND_FILTER = True            # Enable/disable the filter
# SMA_FAST_PERIOD       = 10
# SMA_SLOW_PERIOD       = 100
# SMA_SIGNAL_PERIOD     = 3
# SMA_RANGE_FILTER_PERIOD = 100
# SMA_RANGE_FILTER_MULTIPLIER = 1.5
# # SMA_RANGE_FILTER_MULTIPLIER = 0.5
# SMA_REQUIRE_ALIGNMENT = True           # If True, require pattern direction to match oscillator colour

# ---------------------------------------------------------------------------
# Trend Structure settings (mirror MQL5)
# ---------------------------------------------------------------------------
PIVOT_LENGTH               = 5
REQUIRED_PIVOT_PAIRS       = 2
HOLD_LAST_TREND_ON_UNDEFINED = False
TREND_ENGINE               = "CHOCH_THEN_BOS"
USE_HTF_MAPPING            = False
FILTER_HTF_ALIGNMENT       = False
# ---------------------------------------------------------------------------
# PD Zones: HTF Zone Settings
# ---------------------------------------------------------------------------
INP_USE_HTF_ZONES = False

# ---------------------------------------------------------------------------
# Pattern settings (MQL5-exact)
# ---------------------------------------------------------------------------
SHOW_PATTERN_LABELS          = True
PATTERN_REQUIRE_HTF_AGREEMENT = True
PATTERN_BULL_COLOR            = "LimeGreen"
PATTERN_BEAR_COLOR            = "Red"
PATTERN_FONT_SIZE             = 8
PATTERN_DOJI_BODY_RATIO       = 0.1
PATTERN_SPINNING_TOP_BODY_RATIO = 0.3
PATTERN_LONG_BODY_RATIO       = 0.6
PATTERN_HAMMER_WICK_RATIO     = 2.0    # Min long-wick/body ratio for Hammer/Shooting Star
PATTERN_HAMMER_BODY_RATIO     = 0.35   # Max body/range ratio for Hammer/Shooting Star

# ---------------------------------------------------------------------------
# PD Zones: Range Settings
# ---------------------------------------------------------------------------
INP_LOOKBACK_PERIOD      = 15
INP_BUFFER_POINTS        = 10.0
INP_USE_ATR              = True
INP_ATR_PERIOD           = 5
INP_ATR_MULTIPLIER       = 1.0

# ---------------------------------------------------------------------------
# PD Zones: Manual Zone Boundaries
# ---------------------------------------------------------------------------
INP_USE_MANUAL_ZONES        = False
INP_DISCOUNT_INNER_OFFSET   = 25.0
INP_DISCOUNT_OUTER_OFFSET   = 43.0
INP_PREMIUM_INNER_OFFSET    = 25.0
INP_PREMIUM_OUTER_OFFSET    = 43.0


# ---------------------------------------------------------------------------
# Zone-Interaction Candle Filter
# ---------------------------------------------------------------------------
INP_REQUIRE_ZONE_TOUCH   = True
INP_ZONE_OVERLAP_PERCENT = 10.0


# ---------------------------------------------------------------------------
# Trade cost model
# ---------------------------------------------------------------------------
USE_COMMISSION = False
COMMISSION_R = 0.1

# ---------------------------------------------------------------------------
# Dollar conversion (cosmetic)
# ---------------------------------------------------------------------------
DOLLAR_PER_RISK = 3.0

# ---------------------------------------------------------------------------
# Pattern library selection (not used by Japanese detector)
# ---------------------------------------------------------------------------
PATTERN_LIBRARY = "japanese"

# ---------------------------------------------------------------------------
# Pattern filter (optional – set to None to keep all patterns)
# ---------------------------------------------------------------------------
PATTERN_FILTER = None
# PATTERN_FILTER = [
#     "Piercing",
#     "DarkCloudCover",
#     "BullishEngulfing",
#     "BearishEngulfing",
#     "CustomPiercing",
#     "CustomDarkCloud",
#     "Hammer",
#     "ShootingStar",
# ]
PATTERN_EXCLUDE = [   
    "BullishEngulfing",
    "BearishEngulfing"
    ]
PATTERN_EXCLUDE = None

# ---------------------------------------------------------------------------
# Trade simulator
# ---------------------------------------------------------------------------
RUN_TRADE_SIMULATION = True
OVERLAPPING_TRADES   = True
USE_OPEN_SL = True
# USE_OPEN_SL = False


WARMUP_BARS = 500
TRAILING_BUFFER_BARS = 500



# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
SHOW_OUTCOME_R      = True
SHOW_DAILY_BREAKDOWN = False
RESULT_DIR = "result"
RESULT_FILENAME = f"{SYMBOL}.json"
CONSOLE_LOG_LEVEL = "detail"

# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
DEBUG_PATTERN_SKIPS = False   # set True to print why patterns are skipped