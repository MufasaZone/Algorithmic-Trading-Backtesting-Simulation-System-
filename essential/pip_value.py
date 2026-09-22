import MetaTrader5 as mt5
import json

# -------------------------------
# 1. Connect to the MT5 terminal
# -------------------------------
if not mt5.initialize():
    print("❌ Connection failed")
    quit()

# -------------------------------
# 2. Helper function: pip size
# -------------------------------
def get_pip_size(symbol_info):
    digits = symbol_info.digits
    if symbol_info.trade_calc_mode == mt5.SYMBOL_CALC_MODE_FOREX:
        if digits == 5 or digits == 3:
            return symbol_info.point * 10
        elif digits == 4 or digits == 2:
            return symbol_info.point
        else:
            return symbol_info.point * 10
    else:
        return symbol_info.point * 10

# -------------------------------
# 3. Select symbols & gather data
# -------------------------------
symbols = mt5.symbols_get()
if symbols is None or len(symbols) == 0:
    print("❌ No symbols found")
    mt5.shutdown()
    quit()

json_data = []
account_currency = mt5.account_info().currency if mt5.account_info() else "N/A"

for sym in symbols:
    # Ensure symbol is selected in Market Watch
    if not sym.visible:
        mt5.symbol_select(sym.name, True)

    info = mt5.symbol_info(sym.name)
    if info is None:
        continue   # silently skip symbols we cannot read

    pip_size = get_pip_size(info)
    json_data.append({
        "name": info.name,
        "pip": round(pip_size, 5),
        "account_currency": account_currency
    })

# -------------------------------
# 4. Save JSON & print result
# -------------------------------
with open("pip_value.json", "w") as f:
    json.dump(json_data, f, indent=4)

print(f"✅ Data fetched and saved ({len(json_data)} symbols)")

# -------------------------------
# 5. Shut down connection
# -------------------------------
mt5.shutdown()