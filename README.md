# AlgoTrader 2025

A hybrid ML + Q-Learning trading bot for stocks and crypto. Combines XGBoost classification, technical indicators, news sentiment analysis, and reinforcement learning with live paper trading via Alpaca.

> **Paper trading only by default.** Do not use with real funds without thorough evaluation.

## Features

- **ML Signals** - XGBoost classifier trained on daily OHLCV with technical indicators
- **Q-Learning** - Reinforcement learning layer that refines buy/sell/hold decisions
- **Sentiment Analysis** - News headlines scored via Claude or Grok AI
- **Confidence Scoring** - Multi-source signal aggregation (ML, sentiment, technicals, fundamentals)
- **Crypto Trading** - BTC, ETH, SOL support via Alpaca
- **Risk Management** - Stop loss, take profit, trailing stops, PDT compliance, VIX gating
- **GUI Dashboard** - macOS menu bar app with real-time positions, logs, settings, backtest, and AI chat
- **Dexter AI Agent** - Autonomous financial research assistant for ticker selection and bias analysis

## Quick Start

### 1. Clone and set up Python

```bash
git clone <repo-url> AlgoTrader_2025
cd AlgoTrader_2025
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. (Optional) Install TA-Lib for additional indicators

```bash
brew install ta-lib
pip install TA-Lib
```

### 3. Configure API keys

Copy the example config and add your keys:

```bash
cp config.yaml config.yaml.backup  # optional backup
```

Edit `config.yaml` and replace all `YOUR_*` placeholders:

| Key | Where to get it |
|-----|----------------|
| `alpaca.api_key` / `api_secret` | [alpaca.markets](https://alpaca.markets) (Paper trading account) |
| `polygon.api_key` | [polygon.io](https://polygon.io) (Free tier works) |
| `newsapi_token` | [newsapi.org](https://newsapi.org) |
| `grok_api_key` | [x.ai](https://x.ai) (xAI API) |
| `claude.api_key` | [anthropic.com](https://console.anthropic.com) |

### 4. (Optional) Set up Dexter AI agent

```bash
cd dexter
cp env.example .env
# Edit .env with your API keys
bun install
cd ..
```

### 5. Run the bot

**Option A: Unified launcher (recommended)**

```bash
python trading_bot_app.py
```

This starts the bot service and opens the GUI dashboard. Use the menu bar icon to control everything.

**Option B: macOS Desktop shortcut**

Double-click `Trading Bot.command` on your Desktop (see Setup below).

### 6. First run - train the model

On first launch, the bot needs to train its ML model. From the GUI:
1. The bot will auto-download market data from Polygon
2. It will train the XGBoost model and save to `artifacts/`
3. Subsequent runs will load the existing model

To retrain manually, delete `artifacts/final_model.pkl` and restart.

## Desktop Shortcut (macOS)

Create a file on your Desktop called `Trading Bot.command`:

```bash
#!/bin/bash
cd /path/to/AlgoTrader_2025
source .venv/bin/activate
PYTHON=$(which python3)
nohup $PYTHON trading_bot_app.py > /dev/null 2>&1 &
osascript -e 'tell application "Terminal" to close front window' &
exit
```

Then make it executable: `chmod +x ~/Desktop/Trading\ Bot.command`

## Project Structure

```
AlgoTrader_2025/
├── trading_bot_app.py          # Unified launcher (start here)
├── Trader_main_Grok4_20250731.py  # Core trading engine
├── bot_service.py              # Background service wrapper
├── config.yaml                 # Configuration (API keys, tickers, risk params)
├── requirements.txt            # Python dependencies
│
├── signal_confidence.py        # Multi-source confidence scoring
├── finviz_enrichment.py        # Fundamental data from FinViz
├── pdt_guard.py                # Pattern Day Trader compliance
├── dexter_gate.py              # AI trade veto gate
├── position_monitor.py         # Stop loss enforcement
├── ipc_protocol.py             # GUI <-> bot communication
│
├── gui/                        # PyQt6 GUI application
│   ├── main_window.py          # Tabbed main window
│   ├── menu_bar.py             # macOS menu bar / tray icon
│   ├── dashboard_window.py     # Live positions & orders
│   ├── settings_window.py      # Config editor
│   ├── backtest_window.py      # Backtest runner
│   ├── logs_window.py          # Log viewer
│   ├── chat_window.py          # AI chat (Dexter/Claude)
│   └── manual_trade_dialog.py  # Manual trade entry
│
├── dexter/                     # AI research agent (TypeScript/Bun)
│   ├── src/                    # Agent source code
│   ├── package.json            # Node dependencies
│   └── env.example             # API key template
│
├── artifacts/                  # Generated: ML models, Q-table, reports
├── data/                       # Generated: daily OHLCV CSVs
└── logs/                       # Generated: bot and trade logs
```

## How It Works

1. **Data** - Daily OHLCV pulled from Polygon. Latest bars from Alpaca during live trading.
2. **Features** - Technical indicators (MA, RSI, MACD, Bollinger, ATR, StochRSI), volume changes, lags.
3. **Sentiment** - News headlines classified by Claude/Grok as positive/negative/neutral per ticker.
4. **Model** - XGBoost classifier predicts next-day up/down. Tuned with TimeSeriesSplit cross-validation.
5. **Q-Learning** - Reinforcement layer refines signals based on recent action outcomes.
6. **Confidence** - Aggregates ML, sentiment, technicals, fundamentals, and Dexter bias into a 0-1 score.
7. **Execution** - Places bracket orders (entry + stop loss + take profit) via Alpaca paper API.
8. **Schedule** - Runs at 10:00 AM ET (morning gap trading) and 3:15 PM ET (EOD signals).

## Trading Schedule

| Time (ET) | Action |
|-----------|--------|
| 10:00 AM | Morning run - gap trading signals |
| 3:15 PM | Afternoon run - end-of-day signals |

The bot also monitors positions continuously for stop loss enforcement.

## GUI Features

- **Dashboard** - Live account value, positions, recent orders, P&L
- **Logs** - Real-time bot log viewer and trade history
- **Settings** - Edit all config values, tickers, risk params
- **Backtest** - Run and visualize strategy backtests
- **Chat** - Talk to Dexter AI or get Claude analysis
  - "Ask Claude" - Get Claude's analysis of Dexter's recommendations
  - "Use Claude Tickers" - Get fresh stock & crypto picks from Claude
  - "Use Dexter Tickers" - Get Dexter's ticker recommendations

## Risk Parameters

| Setting | Default | Description |
|---------|---------|-------------|
| `risk_per_trade_pct` | 0.15 | Risk per trade as % of portfolio |
| `stop_loss_pct` | 4% | Stop loss trigger |
| `take_profit_pct` | 8% | Take profit trigger |
| `max_position_pct` | 5% | Max single position size |
| `vix_threshold` | 25 | Skip trades if VIX above this |
| `buying_power_pct` | 50 | % of buying power to use |
| `trail_pct` | 3% | Trailing stop distance |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| TA-Lib warning | Install via `brew install ta-lib && pip install TA-Lib`, or ignore (runs without it) |
| 401 Unauthorized | Check API keys in `config.yaml`. Paper vs live URLs must match account type. |
| 429 Too Many Requests | Reduce tickers or increase `job_interval_minutes` |
| No model found | Run training first (delete `artifacts/final_model.pkl` to force retrain) |
| GUI won't start | Ensure PyQt6 is installed: `pip install PyQt6` |
| Dexter not working | Run `cd dexter && bun install` and check `.env` keys |

## Disclaimer

This software is for educational and research purposes. Markets involve substantial risk. Past performance does not guarantee future results. Use paper trading and validate thoroughly before considering any real capital.
