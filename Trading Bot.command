#!/bin/bash
# AlgoTrader 2025 - Desktop Launcher
# Update the path below to match your installation directory

INSTALL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$INSTALL_DIR"

source .venv/bin/activate
PYTHON=$(which python3)
nohup $PYTHON trading_bot_app.py > /dev/null 2>&1 &
osascript -e 'tell application "Terminal" to close front window' &
exit
