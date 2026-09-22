"""
config.py -- HTF Pattern Scanner configuration

All tunable behaviour lives here. No hidden defaults.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
SYMBOL = "Volatility 15 (1s) Index.0_M1"

_CONFIG_DIR = Path(__file__).resolve().parent
CSV_PATH = str(_CONFIG_DIR / "../.." / "CSV_deriv_zero_spread" / "csv" / f"{SYMBOL}.csv")

CTF_TIMEFRAME = "5min"

# ── Time window – completely removed (empty strings disable filtering) ──
START_DATE = "2026-04-01"
END_DATE   = "2026-04-02"
WARMUP_BARS = 300
TRAILING_BUFFER_BARS = 300
# ---------------------------------------------------------------------------
# Indicator – Trend Structure
# ---------------------------------------------------------------------------
PIVOT_LENGTH               = 20
REQUIRED_PIVOT_PAIRS       = 2
HOLD_LAST_TREND_ON_UNDEFINED = False
TREND_ENGINE               = "CHOCH_THEN_BOS"   # "PIVOT_PAIRS", "BOS_CHOCH", "CHOCH_THEN_BOS"
USE_HTF_MAPPING            = True          # auto‑map CTF → HTF using the standard MQL5 mapping

# ── Trend alignment filter OFF ──
FILTER_HTF_ALIGNMENT       = False

# ---------------------------------------------------------------------------
# Indicator – Pattern Labels
# ---------------------------------------------------------------------------
SHOW_PATTERN_LABELS          = True
PATTERN_REQUIRE_HTF_AGREEMENT = False      # pattern HTF alignment filter OFF
PATTERN_BULL_COLOR            = "LimeGreen"
PATTERN_BEAR_COLOR            = "Red"
PATTERN_FONT_SIZE             = 8
PATTERN_DOJI_BODY_RATIO       = 0.1
PATTERN_SPINNING_TOP_BODY_RATIO = 0.3
PATTERN_LONG_BODY_RATIO       = 0.6

# ---------------------------------------------------------------------------
# Indicator – PD Zones (Range Settings)
# ---------------------------------------------------------------------------
INP_LOOKBACK_PERIOD      = 15
INP_BUFFER_POINTS        = 10.0
INP_USE_ATR              = True
INP_ATR_PERIOD           = 5
INP_ATR_MULTIPLIER       = 1.0

# ---------------------------------------------------------------------------
# Indicator – PD Zones (Manual Zone Boundaries)
# ---------------------------------------------------------------------------
INP_USE_MANUAL_ZONES        = False
INP_DISCOUNT_INNER_OFFSET   = 25.0
INP_DISCOUNT_OUTER_OFFSET   = 43.0
INP_PREMIUM_INNER_OFFSET    = 25.0
INP_PREMIUM_OUTER_OFFSET    = 43.0

# ---------------------------------------------------------------------------
# Indicator – PD Zones (HTF Zone Settings)
# ---------------------------------------------------------------------------
INP_USE_HTF_ZONES = True

# ---------------------------------------------------------------------------
# Indicator – Zone‑Interaction Candle Filter
# ---------------------------------------------------------------------------
# ── Zone‑touch requirement OFF ──
INP_REQUIRE_ZONE_TOUCH   = False

INP_ZONE_OVERLAP_PERCENT = 10.0      # irrelevant when touch is not required

# ---------------------------------------------------------------------------
# Trade parameters
# ---------------------------------------------------------------------------
RR = 4.5
USE_BREAKEVEN = False
BREAKEVEN_TRIGGER = 1.0
BREAKEVEN_BUFFER = 0.1

# ---------------------------------------------------------------------------
# ATR Trailing Stop (optional – if used, fixed TP/RR is ignored)
# ---------------------------------------------------------------------------
USE_ATR_TRAIL   = True
ATR_TRAIL_TRIGGER = 1.1
ATR_TRAIL_PERIOD  = 10
ATR_TRAIL_MULT    = 2.0

# ---------------------------------------------------------------------------
# Trade cost model
# ---------------------------------------------------------------------------
USE_COMMISSION = False
COMMISSION_R = 0.1

# ---------------------------------------------------------------------------
# Dollar conversion (purely cosmetic)
# ---------------------------------------------------------------------------
DOLLAR_PER_RISK = 3.0

# ---------------------------------------------------------------------------
# Pattern library selection
# ---------------------------------------------------------------------------
# ── Include both libraries (all patterns) ──
PATTERN_LIBRARY = "both"

# ---------------------------------------------------------------------------
# Pattern filter (optional – set to None to keep all patterns)
# ---------------------------------------------------------------------------
PATTERN_FILTER = None          # no pattern name filter

# NOTE: no longer used. Left here for compatibility.
PATTERN_PIERCING_DARKCLOUD_REQUIRE_WICK_UNDERCUT = False

# ---------------------------------------------------------------------------
# Trade simulator
# ---------------------------------------------------------------------------
RUN_TRADE_SIMULATION = True

# ── Allow overlapping trades (no overlap filter) ──
OVERLAPPING_TRADES   = True
USE_OPEN_SL = False

# ---------------------------------------------------------------------------
# Output – extra columns & breakdowns
# ---------------------------------------------------------------------------
SHOW_OUTCOME_R      = True
SHOW_DAILY_BREAKDOWN = False

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULT_DIR = "result"
RESULT_FILENAME = f"{SYMBOL}.json"
CONSOLE_LOG_LEVEL = "detail"     # "detail" or "summary"