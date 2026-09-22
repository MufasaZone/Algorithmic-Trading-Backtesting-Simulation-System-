"""
Configuration for TrendStructure + PD Zones offline scanner.
Mirrors the input parameters of the original MQL5 indicator.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
SYMBOL = "Volatility 25 (1s) Index.0_M1"

_CONFIG_DIR = Path(__file__).resolve().parent
CSV_PATH = str(_CONFIG_DIR / ".." / "CSV_deriv_zero_spread" / "csv" / f"{SYMBOL}.csv")

CSV_DATE_FORMAT = None

CHART_TIMEFRAME = "M5"

# SCAN_START = "2025-06-21 00:00:00"
# SCAN_END   = "2025-06-21 23:55:00"


# SCAN_START = "2025-02-01"
# SCAN_END   = "2025-02-10"


SCAN_START = "2024-01-01"
SCAN_END   = "2024-12-30"


# --- Warm-up Gate ---
# Hard-stops the scan if trend/DMSI/zone indicators don't have enough real
# history before SCAN_START to be past their cold-start state. Set to None
# to use the code-derived minimum (pivot_length*2+5 HTF bars for the trend
# engine, lookback*2+10 CTF bars for DMSI). Set to an int to require at
# least that many HTF bars of lead-in on top of the derived minimum.
WARMUP_HTF_BARS = 300

# --- Debug / Output ---
# When False, skips building the large per-bar JSON array (~20-26% of
# runtime on big scans). Summary counts and all other report sections are
# unaffected either way.
SHOW_BARS_DEBUG = True


# --- Trend Structure Settings ---
PivotLength               = 10
RequiredPivotPairs        = 2
HoldLastTrendOnUndefined  = False
ShowAlerts                = False

TrendEngine                = "choch_then_bos"

ShowPivotLabels            = False
UseHTFMapping               = False
FilterHTFAlignment          = False

# --- Trend Age / Distance ---
ShowTrendAgeInfo           = True

# --- Pattern Labels ---
ShowPatternLabels           = True
PatternRequireHtfAgreement  = True
PatternFontSize              = 8
PatternDojiBodyRatio         = 0.1
PatternSpinningTopBodyRatio  = 0.3
PatternLongBodyRatio         = 0.6

# All patterns the scanner recognizes (mirrors main.py's PATTERN_NAMES,
# same order/indices as the original MQL5 indicator):
#   "Piercing", "DarkCloudCover", "BullishEngulfing", "BearishEngulfing",
#   "CustomPiercing", "CustomDarkCloud", "MorningStar", "EveningStar",
#   "DojiConfirmedBull", "DojiConfirmedBear",
#   "SpinTopConfirmedBull", "SpinTopConfirmedBear"
#
# Set ENABLED_PATTERNS to a list of the names above to scan only those
# patterns. Leave it as None to scan all 12 (matches the original
# indicator, which has no such filter -- this is scanner-only).

ENABLED_PATTERNS = ["SpinTopConfirmedBear", "SpinTopConfirmedBull"]
ENABLED_PATTERNS = None

# Set EXCLUDE_PATTERNS to a list of the names above to drop those patterns
# from whatever ENABLED_PATTERNS would otherwise scan (applied after
# ENABLED_PATTERNS -- works whether that's None/all or an explicit list).
# Leave it as None or [] to exclude nothing.
EXCLUDE_PATTERNS = ["SpinTopConfirmedBear", "SpinTopConfirmedBull", "DojiConfirmedBull", 
                    "EveningStar", "MorningStar", "DojiConfirmedBear"
                    ]

EXCLUDE_PATTERNS = ["SpinTopConfirmedBear", "SpinTopConfirmedBull"]


# EXCLUDE_PATTERNS = None


# --- PD Zones: Range Settings ---
InpLookbackPeriod   = 15
InpBufferPoints     = 10
InpUseATR           = True
InpATRPeriod        = 5
InpATRMultiplier    = 1.0

# --- PD Zones: Manual Zone Boundaries ---
InpUseManualZones       = False
InpDiscountInnerOffset  = 25.0
InpDiscountOuterOffset  = 43.0
InpPremiumInnerOffset   = 25.0
InpPremiumOuterOffset   = 43.0

# --- PD Zones: HTF Zone Settings ---
InpUseHTFZones = True

# --- PD Zones: Signal Settings ---
InpShowSignals         = False
InpConfluenceLookback   = 10
InpMinImpulsePoints     = 15

# --- Zone-Interaction Candle Filter ---
InpRequireZoneTouch     = True
InpZoneOverlapPercent   = 10.0

# --- DMSI Regime Filter ---
InpUseDMSIFilter        = True
# InpUseDMSIFilter        = False
InpDMSILookback         = 20
InpDMSIPercentilePeriod = 80
InpDMSITrendPercentile  = 80
InpDMSIRangePercentile  = 25
InpDMSIFastSmooth       = 2
InpDMSISlowSmooth       = 40

# --- Symbol point size (min price increment); adjust to your data ---
POINT_SIZE = 0.01

# --- Trade Simulation Settings ---
RR                  = 10.0
# USE_BREAKEVEN       = True
USE_BREAKEVEN       = False
BREAKEVEN_TRIGGER   = 2.5   # move to BE once price reaches this many R in favor
BREAKEVEN_BUFFER    = 0.1   # BE stop = entry +/- this many R (0 = exact entry)

# Whether trailing is used at all is controlled by TRAIL_TYPE below
# (set it to "OFF" for a fixed-TP-only baseline) -- there is no separate
# on/off switch here anymore.
# ATR_TRAIL_TRIGGER   = 1.0   # arm trailing once price reaches this many R in favor
ATR_TRAIL_TRIGGER   = 2.0   # arm trailing once price reaches this many R in favor
# ATR_TRAIL_PERIOD    = 14
# ATR_TRAIL_MULT      = 1.5

ATR_TRAIL_PERIOD    = 14
ATR_TRAIL_MULT      = 4

# --- Stepped ATR Trail ---
# ATR_TRAIL_PERIOD/MULT above still control the ARMING behavior (when the
# trail first engages, at ATR_TRAIL_TRIGGER R, using the slow period-50
# ATR). Once armed, trail DISTANCE switches to this faster ATR so it
# reacts to current volatility instead of a stale ~4hr-wide average, and
# the multiplier tightens in two steps as the trade's max favorable R
# extends further -- wide early (rides out normal pullback noise right
# after arming), progressively tighter once the move is clearly extended
# (locks in more of the peak instead of giving most of it back).
ATR_TRAIL_FAST_PERIOD = 14   # ATR period used for trail distance once armed
ATR_TRAIL_STEP2_R      = 15.0  # once max favorable R reaches this, tighten mult
ATR_TRAIL_STEP2_MULT   = 3.0
ATR_TRAIL_STEP3_R      = 20.0  # once max favorable R reaches this, tighten further
ATR_TRAIL_STEP3_MULT   = 2.0

# --- Trail Type Selector ---
# Choose which trailing algorithm drives the stop once trailing arms
# (arming itself always uses ATR_TRAIL_TRIGGER R on the slow ATR, shared
# by every type below so they're comparable). Valid values:
#
#   "OFF"              -- no trailing; RR (fixed take-profit) is the only
#                          way a winner exits. Useful as a baseline.
#   "STEPPED_ATR"       -- fast-ATR distance, multiplier tightens through
#                          ATR_TRAIL_STEP2_*/STEP3_* as the trade extends.
#                          (what the last several runs have been using)
#   "CHANDELIER_ATR"    -- fast-ATR distance anchored to the best high/low
#                          reached so far in the trade (not just this
#                          bar's), so one spike bar can't threaten the
#                          trail on the very next ordinary pullback bar.
#   "STRUCTURE_SWING"   -- trail behind the last CONFIRMED pivot low
#                          (bull) / pivot high (bear), using the same
#                          PivotLength as the trend engine, plus a small
#                          ATR buffer. Room scales with actual market
#                          structure instead of a volatility distance.
#   "R_RATCHET"         -- deterministic step to a locked R level (no ATR
#                          at all) the first time max favorable R crosses
#                          each RATCHET_TRIGGER_R threshold below.
#   "PARABOLIC"         -- SAR-style continuous acceleration: trail
#                          distance shrinks smoothly as the trade makes
#                          new favorable extremes, instead of jumping in
#                          discrete steps.
#   "TIME_BASED"        -- fast-ATR multiplier shrinks linearly from
#                          TIME_TRAIL_MULT_START to _END purely as a
#                          function of bars-since-entry, independent of
#                          price/ATR magnitude.
#   "DMSI_REGIME"       -- fast-ATR multiplier is wide while DMSI reads
#                          "trending" at the current bar and snaps tight
#                          the moment it doesn't -- ties the trail to the
#                          same regime read already used for entries.
TRAIL_TYPE = "STEPPED_ATR"

# --- Chandelier / structure-swing shared buffer ---
# Extra distance (in units of fast ATR) added beyond the raw swing level
# for STRUCTURE_SWING, so the stop sits just past the pivot rather than
# exactly on it.
SWING_TRAIL_BUFFER_MULT = 0.5

# --- R_RATCHET steps ---
# Parallel lists: the first time max favorable R reaches
# RATCHET_TRIGGER_R[k], current_sl locks to entry +/- RATCHET_LOCK_R[k]*risk
# (in the trade's favor). Must be the same length; keep both ascending.
RATCHET_TRIGGER_R = [3.0, 6.0, 10.0, 15.0]
RATCHET_LOCK_R    = [1.0, 4.0, 8.0, 12.0]

# --- PARABOLIC (SAR-style) settings ---
PARABOLIC_AF_START = 0.10   # starting acceleration factor
PARABOLIC_AF_STEP  = 0.05   # increment applied on each new favorable extreme
PARABOLIC_AF_MAX   = 0.60   # cap on acceleration factor (higher = tighter)

# --- TIME_BASED settings ---
TIME_TRAIL_BARS_FULL_TIGHTEN = 100   # bars-since-entry to reach the tightest mult
TIME_TRAIL_MULT_START = 6.0          # multiplier at bar 0 (right after arming)
TIME_TRAIL_MULT_END   = 2.0          # multiplier once TIME_TRAIL_BARS_FULL_TIGHTEN elapses

# --- DMSI_REGIME settings ---
DMSI_TRAIL_WIDE_MULT  = 5.0   # multiplier while DMSI regime == trending
DMSI_TRAIL_TIGHT_MULT = 2.0   # multiplier otherwise (transition/ranging)


OVERLAPPING_TRADES  = True   # allow new hits while a previous trade is still open
USE_OPEN_SL         = False  # if True, SL exit uses bar close instead of intrabar low/high

USE_COMMISSION      = False
COMMISSION_R        = 0.0    # flat R deducted per closed trade

SHOW_OUTCOME_R      = True
SHOW_DAILY_BREAKDOWN = False

RESULTS_DIR = "results"
RESULTS_JSON_NAME = None