# Algorithmic Trading Backtesting & Simulation System

A Python backtesting and trade-simulation engine for synthetic market indices (Deriv Volatility and High Frequency indices), built by porting and extending core logic from an original MQL5 (MetaTrader 5) trading indicator.

The system detects market structure, filters signals through a custom regime index, identifies premium/discount zones, recognizes candlestick patterns, and simulates trade outcomes across historical price data — all driven by a single configuration file so strategies can be swept across multiple symbols and timeframes without touching the core logic.

## What it does

- **Market structure detection** — identifies trend direction using pivot-based Break of Structure (BOS) / Change of Character (CHoCH) logic, ported from the original MQL5 indicator's trend engine
- **Dynamic Market Structure Index (DMSI)** — a custom-built regime filter that classifies market conditions (trending vs. ranging) to filter out low-quality signals
- **SMA Trend Strength Oscillator** — an optional secondary trend-alignment filter
- **Premium/Discount (PD) zone detection** — identifies high-probability reversal zones using ATR-based or manual range calculation
- **Candlestick pattern recognition** — detects Japanese candlestick patterns (engulfing, hammer, shooting star, piercing, dark cloud cover, and others), matched exactly to the original MQL5 pattern-detection logic
- **Trade simulation engine** — simulates full trade lifecycles including stop-loss/take-profit execution, breakeven management, R-multiple outcome tracking, and overlapping trade handling
- **Performance reporting** — generates equity curve analysis, drawdown tracking, and detailed performance statistics per run
- **Multi-symbol / multi-timeframe sweeps** — runs the full pipeline across any combination of symbols and timeframes defined in the config, and prints a comparison matrix of results

## Project structure

```
config.py / v2_config.py   Configuration — symbols, timeframes, trade parameters,
                            filter settings, date ranges (single source of truth
                            for a given run)
main.py / v2_main.py       Core engine — trend detection, PD zones, DMSI, pattern
                            detection, trade simulation, performance reporting
result/                    JSON output per symbol/timeframe run
Raw/                       Raw input data
essential/, old/           Supporting and earlier-version files
```

## How it works

1. Set the symbols, timeframes, and trade parameters (risk/reward, stop-loss buffer, filters) in the config file.
2. Run the main script — it loads historical OHLC data, computes trend structure, PD zones, and any enabled filters, detects candlestick patterns, and simulates trades bar-by-bar.
3. Results are written to `result/` as JSON, and a comparison matrix is printed across every symbol/timeframe combination in the sweep.

## Background

This project started as a direct Python port of an MQL5 indicator (`TrendStructure_PDZones`) originally built for MetaTrader 5, with the goal of running large-scale historical backtests outside the constraints of the MetaTrader strategy tester. Beyond the port, it extends the original logic with additional filters (DMSI, SMA oscillator) and a full trade simulation and reporting layer.

## Status

Actively developed, used for personal strategy research and backtesting on synthetic indices.
