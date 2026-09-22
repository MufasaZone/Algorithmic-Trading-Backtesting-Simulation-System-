"""
config.py -- TrendStructure + PDZones filter + MQL5-exact Japanese pattern scanner + Trade Simulator
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
# The scanner runs once per (symbol, timeframe) pair drawn from
# SYMBOLS x TIMEFRAMES (each writes its own result/{symbol}_{timeframe}.json)
# and prints a symbol x timeframe matrix comparison at the end.
SYMBOLS = [
    # "Volatility 5 (1s) Index.0_M1",
    # "Volatility 10 (1s) Index.0_M1",
    # "Volatility 15 (1s) Index.0_M1",
    # "Volatility 25 (1s) Index.0_M1",
    # "Volatility 30 (1s) Index.0_M1",
    # "Volatility 50 (1s) Index.0_M1",
    # "Volatility 75 (1s) Index.0_M1",


    # "High Frequency Vol 25 Index.0_M1",
    "High Frequency Vol 50 Index.0_M1",
]

# TIMEFRAMES = [
#     "3min",
#     "4min",
#     "5min",
#     "6min",
#     # "15min",
#     # "30min",
# ]

TIMEFRAMES = [
    # "3min",
    "5min",
    # "10min",
    # "30min",
]


_CONFIG_DIR = Path(__file__).resolve().parent

def csv_path_for(symbol: str) -> str:
    """Resolve the CSV path for any symbol using the same layout as CSV_PATH."""
    return str(_CONFIG_DIR / ".." / "CSV_deriv_zero_spread" / "csv" / f"{symbol}.csv")

# Per-symbol SL buffer (% of entry price added beyond the raw SL extreme).
# Every entry in SYMBOLS must have a key here -- no fallback/default.
SL_BUFFER_PCT_BY_SYMBOL = {
    # "Volatility 5 (1s) Index.0_M1": 0.00,
    "Volatility 5 (1s) Index.0_M1": 0.005,
    # "Volatility 10 (1s) Index.0_M1": 0.0,    
    "Volatility 10 (1s) Index.0_M1": 0.02,

    "Volatility 15 (1s) Index.0_M1": 0.02,
    "Volatility 25 (1s) Index.0_M1": 0.03,
    "Volatility 30 (1s) Index.0_M1": 0.03,
    "Volatility 50 (1s) Index.0_M1": 0.03,
    "Volatility 75 (1s) Index.0_M1": 0.03,

    "High Frequency Vol 25 Index.0_M1": 0.07,
    "High Frequency Vol 50 Index.0_M1": 0.07,
}

def sl_buffer_pct_for(symbol: str) -> float:
    """Look up the SL buffer % for a symbol. Raises KeyError if missing --
    every symbol in SYMBOLS must have an explicit entry."""
    return SL_BUFFER_PCT_BY_SYMBOL[symbol]

# Working values -- main() overwrites these per (symbol, timeframe) iteration
# during the sweep. They start out pointing at the first entry in each list
# purely so anything importing this module before main() runs (or a helper
# called standalone) still has a valid symbol/timeframe/path/SL-buffer to read.
SYMBOL = SYMBOLS[0]
CTF_TIMEFRAME = TIMEFRAMES[0]
CSV_PATH = csv_path_for(SYMBOL)
SL_BUFFER_PCT = sl_buffer_pct_for(SYMBOL)


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------

# START_DATE = "2026-03-28"
# END_DATE   = "2026-03-29"

# START_DATE = "2026-08-25"
# END_DATE   = "2026-12-06"


START_DATE = "2025-01-01"
END_DATE   = "2026-12-06"


# ---------------------------------------------------------------------------
# Trade parameters
# ---------------------------------------------------------------------------
# RR = 10
RR = 9

# TP placement mode:
#   "RR"      -> TP = entry +/- RR * risk (original behavior)
#   "PERCENT" -> TP = entry +/- TP_PERCENT% of entry price (fixed price-% target,
#                independent of stop distance/risk)
# TP_MODE = "RR"
TP_MODE = "PERCENT"
# TP_PERCENT = 1   # only used when TP_MODE == "PERCENT" (e.g. 1.0 = 1% away from entry)
# TP_PERCENT = 0.6   # only used when TP_MODE == "PERCENT" (e.g. 1.0 = 1% away from entry)
TP_PERCENT = 0.8   # only used when TP_MODE == "PERCENT" (e.g. 1.0 = 1% away from entry)

TP_PERCENT_MAX_R = 20


# USE_BREAKEVEN = True
USE_BREAKEVEN = False

# BE_MODE = "RR"
BE_MODE = "PERCENT"
# BE_TRIGGER_PERCENT = 0.07   # only used when BE_MODE == "PERCENT": arm BE once price
BE_TRIGGER_PERCENT = 0.45   # only used when BE_MODE == "PERCENT": arm BE once price
                           # moves this % from entry in the favorable direction

# BREAKEVEN_TRIGGER = 2.8
BREAKEVEN_TRIGGER = 3
BREAKEVEN_BUFFER = 0.1
# SL_BUFFER_PCT is set per-symbol above via SL_BUFFER_PCT_BY_SYMBOL / sl_buffer_pct_for()


# ---------------------------------------------------------------------------
# SMA Trend Strength Oscillator Filter
# ---------------------------------------------------------------------------
# VOl 30 (1s)
# USE_SMA_TREND_FILTER = True            # Enable/disable the filter
USE_SMA_TREND_FILTER = False            # Enable/disable the filter
SMA_MA_METHOD          = "SMA"          # "SMA" or "EMA" -- applies to both fast & slow MA (mirrors .mq5 MAMethodType)
# SMA_MA_METHOD          = "EMA"          # "SMA" or "EMA" -- applies to both fast & slow MA (mirrors .mq5 MAMethodType)
SMA_FAST_PERIOD       = 20
# SMA_SLOW_PERIOD       = 77
SMA_SLOW_PERIOD       = 100
SMA_SIGNAL_PERIOD     = 3
SMA_RANGE_FILTER_PERIOD = 80
# SMA_RANGE_FILTER_MULTIPLIER = 2.0
SMA_RANGE_FILTER_MULTIPLIER = 1.3
SMA_REQUIRE_ALIGNMENT = True           # If True, require pattern direction to match oscillator colour

# ---------------------------------------------------------------------------
# Dynamic Market Structure Index (DMSI) Regime Filter
# ---------------------------------------------------------------------------
USE_DMSI_FILTER          =  True
# USE_DMSI_FILTER          =  False
DMSI_LOOKBACK             = 20     # InpLookback
DMSI_PERCENTILE_PERIOD    = 100    # InpPercentilePeriod
DMSI_TREND_PERCENTILE     = 60     # InpTrendPercentile
DMSI_RANGE_PERCENTILE     = 30     # InpRangePercentile
DMSI_ER_WEIGHT            = 0.65   # InpERWeight (clamped 0.55-0.85, same as .mq5)
DMSI_FAST_SMOOTH          = 3      # InpFastSmooth
DMSI_SLOW_SMOOTH          = 35     # InpSlowSmooth

DMSI_SMOOTH_POWER         = 1.15   # InpSmoothPower



# ---------------------------------------------------------------------------
# Trend Structure settings (mirror MQL5)
# ---------------------------------------------------------------------------
PIVOT_LENGTH               = 5
REQUIRED_PIVOT_PAIRS       = 2
HOLD_LAST_TREND_ON_UNDEFINED = False
# TREND_ENGINE               = "BOS_CHOCH"
TREND_ENGINE               = "CHOCH_THEN_BOS"
# TREND_ENGINE               = "PIVOT_PAIRS"
USE_HTF_MAPPING            = False
FILTER_HTF_ALIGNMENT       = False
# Structural confirmation for BOS_CHOCH / CHOCH_THEN_BOS: minimum number of
# opposite-side (BOS_CHOCH: breaking-side-since-last-flip) pivots required
# to confirm before a trend flip is allowed to lock in. 1 = original
# behavior (a single break confirms immediately). Higher = slower to flip,
# fewer fakeout flips, more lag. Mirrors ChochBosMinPivots in the .mq5.
# CHOCH_BOS_MIN_PIVOTS       = 1
CHOCH_BOS_MIN_PIVOTS       = 2
# ---------------------------------------------------------------------------
# PD Zones: HTF Zone Settings
# ---------------------------------------------------------------------------
# INP_USE_HTF_ZONES = True
INP_USE_HTF_ZONES = False

# ---------------------------------------------------------------------------
# Zone-Interaction Candle Filter
# ---------------------------------------------------------------------------
INP_REQUIRE_ZONE_TOUCH   = True
# INP_REQUIRE_ZONE_TOUCH   = False
INP_ZONE_OVERLAP_PERCENT = 10.0


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
    # "CustomDarkCloud", 
    # "CustomPiercing"
    ]

# PATTERN_EXCLUDE = None

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