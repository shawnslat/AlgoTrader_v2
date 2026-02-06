"""
Main Window - Unified tabbed interface for the Trading Bot
Combines Dashboard, Logs, Settings, Backtest, and Chat into one window.
"""
import os
import sys
import threading
from pathlib import Path

import pandas as pd
import subprocess
import yaml

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QTableWidget, QTableWidgetItem, QPushButton,
    QGroupBox, QMessageBox, QButtonGroup, QRadioButton,
    QTextEdit, QLineEdit, QDoubleSpinBox, QCheckBox,
    QComboBox, QSpinBox, QScrollArea
)
from PyQt6.QtCore import QTimer, Qt, QObject, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor, QPixmap

from ipc_protocol import IPCClient
import logging

logger = logging.getLogger(__name__)


def is_crypto_ticker(ticker: str) -> bool:
    """Check if ticker is a cryptocurrency"""
    return '/' in ticker or ticker.endswith('USD') and len(ticker) <= 7


class _ChatSignals(QObject):
    finished = pyqtSignal(dict)
    claude_finished = pyqtSignal(dict)


class MainWindow(QMainWindow):
    """Main unified window with tabs for all trading bot features"""

    def __init__(self, ipc_client: IPCClient):
        super().__init__()
        self.ipc_client = ipc_client
        self.config_path = Path("config.yaml")
        self.config = {}

        # Chat signals
        self.chat_signals = _ChatSignals()
        self.chat_signals.finished.connect(self._on_chat_response)
        self.chat_signals.claude_finished.connect(self._on_claude_response)

        # Store last Dexter response for Claude analysis
        self.last_dexter_response = ""

        # Data
        self.account_data = {}
        self.positions_data = []
        self.signals_data = []
        self.position_filter = "all"

        # Logs state
        base_dir = Path(__file__).resolve().parent.parent
        self.primary_log_file = base_dir / "logs" / "bot_service.log"
        self.fallback_log_file = base_dir / "logs" / "master_trading_bot.log"
        self.log_file = self.primary_log_file
        self.last_log_position = 0

        # Backtest paths
        os.makedirs("artifacts", exist_ok=True)
        self.report_path = Path("artifacts/classification_report.txt")
        self.cm_path = Path("artifacts/confusion_matrix.png")
        self.bt_png_path = Path("artifacts/backtesting_results.png")
        self.bt_csv_path = Path("artifacts/backtesting_results.csv")
        self.validation_report_path = Path("artifacts/validation_report.txt")
        self.oos_comparison_path = Path("artifacts/out_of_sample_comparison.png")
        self.feature_importance_path = Path("artifacts/feature_importance.png")

        self.setup_ui()
        self.load_settings_config()

        # Set initial log position
        if self.log_file.exists() and self.log_file.stat().st_size > 0:
            self.last_log_position = max(0, self.log_file.stat().st_size - 50000)

        # Timers
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_dashboard)
        self.refresh_timer.start(5000)

        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.refresh_logs)
        self.log_timer.start(2000)

        self.backtest_timer = QTimer()
        self.backtest_timer.timeout.connect(self.refresh_backtest_status)
        self.backtest_timer.start(2000)

        # Initial load
        self.refresh_dashboard()
        self.refresh_logs()
        self.load_backtest_artifacts()

    def setup_ui(self):
        """Setup the main window UI with tabs"""
        self.setWindowTitle("Trading Bot")
        self.setGeometry(100, 100, 1200, 800)

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        self.tabs.addTab(self.create_dashboard_tab(), "📊 Dashboard")
        self.tabs.addTab(self.create_logs_tab(), "📄 Logs")
        self.tabs.addTab(self.create_settings_tab(), "⚙️ Settings")
        self.tabs.addTab(self.create_backtest_tab(), "🧪 Backtest")
        self.tabs.addTab(self.create_chat_tab(), "💬 Dexter")

    # ==================== DASHBOARD TAB ====================
    def create_dashboard_tab(self) -> QWidget:
        """Create the dashboard tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Account Summary
        account_group = QGroupBox("Account Summary")
        account_layout = QHBoxLayout()

        self.cash_label = QLabel("Cash: $0.00")
        self.cash_label.setFont(QFont("Arial", 14))
        self.buying_power_label = QLabel("Buying Power: $0.00")
        self.buying_power_label.setFont(QFont("Arial", 14))
        self.portfolio_value_label = QLabel("Portfolio Value: $0.00")
        self.portfolio_value_label.setFont(QFont("Arial", 14))
        self.equity_label = QLabel("Equity: $0.00")
        self.equity_label.setFont(QFont("Arial", 14))

        account_layout.addWidget(self.cash_label)
        account_layout.addWidget(self.buying_power_label)
        account_layout.addWidget(self.portfolio_value_label)
        account_layout.addWidget(self.equity_label)
        account_layout.addStretch()
        account_group.setLayout(account_layout)
        layout.addWidget(account_group)

        # Positions
        positions_group = QGroupBox("Current Positions")
        positions_layout = QVBoxLayout()

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show:"))
        self.filter_button_group = QButtonGroup()

        self.all_radio = QRadioButton("All")
        self.all_radio.setChecked(True)
        self.all_radio.toggled.connect(lambda: self.set_position_filter("all"))
        self.filter_button_group.addButton(self.all_radio)
        filter_layout.addWidget(self.all_radio)

        self.stocks_radio = QRadioButton("📈 Stocks")
        self.stocks_radio.toggled.connect(lambda: self.set_position_filter("stocks"))
        self.filter_button_group.addButton(self.stocks_radio)
        filter_layout.addWidget(self.stocks_radio)

        self.crypto_radio = QRadioButton("🪙 Crypto")
        self.crypto_radio.toggled.connect(lambda: self.set_position_filter("crypto"))
        self.filter_button_group.addButton(self.crypto_radio)
        filter_layout.addWidget(self.crypto_radio)
        filter_layout.addStretch()
        positions_layout.addLayout(filter_layout)

        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(8)
        self.positions_table.setHorizontalHeaderLabels([
            "Type", "Ticker", "Qty", "Entry Price", "Current Price",
            "Market Value", "P&L", "P&L %"
        ])
        self.positions_table.setAlternatingRowColors(True)
        positions_layout.addWidget(self.positions_table)
        positions_group.setLayout(positions_layout)
        layout.addWidget(positions_group)

        # Signals
        signals_group = QGroupBox("Today's Trading Signals")
        signals_layout = QVBoxLayout()
        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(4)
        self.signals_table.setHorizontalHeaderLabels(["Ticker", "Signal", "Confidence", "Time"])
        self.signals_table.setAlternatingRowColors(True)
        signals_layout.addWidget(self.signals_table)
        signals_group.setLayout(signals_layout)
        layout.addWidget(signals_group)

        # Buttons
        buttons_layout = QHBoxLayout()
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_dashboard)
        buttons_layout.addWidget(refresh_btn)

        refresh_signals_btn = QPushButton("🧠 Refresh Signals")
        refresh_signals_btn.clicked.connect(self.refresh_signals_now)
        buttons_layout.addWidget(refresh_signals_btn)

        manual_trade_btn = QPushButton("📊 Manual Trade")
        manual_trade_btn.clicked.connect(self.open_manual_trade)
        buttons_layout.addWidget(manual_trade_btn)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

        return widget

    def refresh_dashboard(self):
        """Refresh dashboard data"""
        self.refresh_account()
        self.refresh_positions()
        self.refresh_signals()

    def refresh_account(self):
        """Refresh account summary"""
        response = self.ipc_client.send_command({"command": "get_account"})
        if response.get("error"):
            return

        self.account_data = response
        self.cash_label.setText(f"Cash: ${response.get('cash', 0):,.2f}")
        self.buying_power_label.setText(f"Buying Power: ${response.get('buying_power', 0):,.2f}")
        self.portfolio_value_label.setText(f"Portfolio Value: ${response.get('portfolio_value', 0):,.2f}")
        self.equity_label.setText(f"Equity: ${response.get('equity', 0):,.2f}")

    def set_position_filter(self, filter_type: str):
        """Set position filter"""
        self.position_filter = filter_type
        self.update_positions_display()

    def refresh_positions(self):
        """Refresh positions"""
        response = self.ipc_client.send_command({"command": "get_positions"})
        if response.get("error"):
            return
        self.positions_data = response.get("positions", [])
        self.update_positions_display()

    def update_positions_display(self):
        """Update positions table"""
        filtered = []
        for pos in self.positions_data:
            ticker = pos.get("ticker", "")
            is_crypto = is_crypto_ticker(ticker)
            if self.position_filter == "all":
                filtered.append(pos)
            elif self.position_filter == "stocks" and not is_crypto:
                filtered.append(pos)
            elif self.position_filter == "crypto" and is_crypto:
                filtered.append(pos)

        self.positions_table.setRowCount(len(filtered))
        for row, pos in enumerate(filtered):
            ticker = pos.get("ticker", "")
            qty = pos.get("qty", pos.get("quantity", 0))
            entry_price = pos.get("avg_entry_price", pos.get("average_price", 0))
            current_price = pos.get("current_price", 0)
            market_value = pos.get("market_value", 0)
            unrealized_pl = pos.get("unrealized_pl", 0)
            unrealized_plpc = pos.get("unrealized_plpc", 0)

            is_crypto = is_crypto_ticker(ticker)
            type_icon = "🪙" if is_crypto else "📈"
            self.positions_table.setItem(row, 0, QTableWidgetItem(type_icon))
            self.positions_table.setItem(row, 1, QTableWidgetItem(ticker))

            qty_str = f"{qty:.8f}".rstrip('0').rstrip('.') if is_crypto else str(int(qty)) if qty == int(qty) else f"{qty:.2f}"
            self.positions_table.setItem(row, 2, QTableWidgetItem(qty_str))
            self.positions_table.setItem(row, 3, QTableWidgetItem(f"${entry_price:.2f}"))
            self.positions_table.setItem(row, 4, QTableWidgetItem(f"${current_price:.2f}"))
            self.positions_table.setItem(row, 5, QTableWidgetItem(f"${market_value:.2f}"))

            pl_item = QTableWidgetItem(f"${unrealized_pl:.2f}")
            pl_pct_item = QTableWidgetItem(f"{unrealized_plpc:.2f}%")
            if unrealized_pl >= 0:
                pl_item.setForeground(Qt.GlobalColor.darkGreen)
                pl_pct_item.setForeground(Qt.GlobalColor.darkGreen)
            else:
                pl_item.setForeground(Qt.GlobalColor.red)
                pl_pct_item.setForeground(Qt.GlobalColor.red)
            self.positions_table.setItem(row, 6, pl_item)
            self.positions_table.setItem(row, 7, pl_pct_item)

        self.positions_table.resizeColumnsToContents()

    def refresh_signals(self):
        """Refresh signals"""
        response = self.ipc_client.send_command({"command": "get_signals"})
        if response.get("error"):
            return
        signals = response.get("signals", [])
        self.signals_data = signals
        self.signals_table.setRowCount(len(signals))

        for row, signal in enumerate(signals):
            self.signals_table.setItem(row, 0, QTableWidgetItem(signal.get("ticker", "")))
            action = signal.get("action", "")
            signal_item = QTableWidgetItem(action.upper())
            if action.lower() == "buy":
                signal_item.setForeground(Qt.GlobalColor.darkGreen)
            elif action.lower() == "sell":
                signal_item.setForeground(Qt.GlobalColor.red)
            self.signals_table.setItem(row, 1, signal_item)
            self.signals_table.setItem(row, 2, QTableWidgetItem(f"{signal.get('confidence', 0):.2f}"))
            self.signals_table.setItem(row, 3, QTableWidgetItem(signal.get("timestamp", "")))

        self.signals_table.resizeColumnsToContents()

    def refresh_signals_now(self):
        """Force signals refresh"""
        response = self.ipc_client.send_command({"command": "refresh_signals"}, timeout=2.0)
        if response.get("error"):
            QMessageBox.warning(self, "Signals", f"Failed: {response.get('error')}")
        elif response.get("success") is False:
            QMessageBox.information(self, "Signals", response.get("message", "Already running."))
        else:
            QMessageBox.information(self, "Signals", "Refresh started.")

    def open_manual_trade(self):
        """Open manual trade dialog"""
        from gui.manual_trade_dialog import ManualTradeDialog
        dialog = ManualTradeDialog(self.ipc_client, self)
        dialog.exec()
        self.refresh_dashboard()

    # ==================== LOGS TAB ====================
    def create_logs_tab(self) -> QWidget:
        """Create the logs tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Controls
        controls_layout = QHBoxLayout()
        self.log_file_label = QLabel(f"File: {self.log_file.name}")
        controls_layout.addWidget(self.log_file_label)

        controls_layout.addWidget(QLabel("Filter:"))
        self.log_filter_combo = QComboBox()
        self.log_filter_combo.addItems(["All", "INFO", "WARNING", "ERROR", "DEBUG"])
        self.log_filter_combo.currentTextChanged.connect(self.apply_log_filter)
        controls_layout.addWidget(self.log_filter_combo)
        controls_layout.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_logs)
        controls_layout.addWidget(clear_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_logs)
        controls_layout.addWidget(refresh_btn)
        layout.addLayout(controls_layout)

        # Log text
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_text.setFontFamily("Courier")
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.log_text)

        return widget

    def refresh_logs(self):
        """Refresh logs"""
        primary_ok = self.primary_log_file.exists() and self.primary_log_file.stat().st_size > 0
        fallback_ok = self.fallback_log_file.exists() and self.fallback_log_file.stat().st_size > 0

        if primary_ok:
            if self.log_file != self.primary_log_file:
                self.log_file = self.primary_log_file
                self.log_file_label.setText(f"File: {self.log_file.name}")
                self.log_text.clear()
                self.last_log_position = 0
        elif fallback_ok:
            if self.log_file != self.fallback_log_file:
                self.log_file = self.fallback_log_file
                self.log_file_label.setText(f"File: {self.log_file.name}")
                self.log_text.clear()
                self.last_log_position = 0
        else:
            self.log_file_label.setText("File: (no logs yet)")
            return

        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(self.last_log_position)
                if self.last_log_position > 0 and self.log_text.toPlainText() == "":
                    f.readline()
                new_lines = f.readlines()
                if new_lines:
                    self.last_log_position = f.tell()
                    for line in new_lines:
                        self.append_log_line(line.rstrip())
                    cursor = self.log_text.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    self.log_text.setTextCursor(cursor)

                size_mb = self.log_file.stat().st_size / (1024 * 1024)
                self.log_file_label.setText(f"File: {self.log_file.name} ({size_mb:.1f} MB)")
        except Exception as e:
            self.log_file_label.setText(f"Error: {e}")

    def append_log_line(self, line: str):
        """Append log line with color"""
        current_filter = self.log_filter_combo.currentText()
        if current_filter != "All" and f"[{current_filter}]" not in line:
            return

        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        if "[ERROR]" in line:
            cursor.insertHtml(f'<span style="color: #f44336;">{line}</span><br>')
        elif "[WARNING]" in line:
            cursor.insertHtml(f'<span style="color: #ffa726;">{line}</span><br>')
        elif "[INFO]" in line:
            cursor.insertHtml(f'<span style="color: #4fc3f7;">{line}</span><br>')
        elif "[DEBUG]" in line:
            cursor.insertHtml(f'<span style="color: #9e9e9e;">{line}</span><br>')
        else:
            cursor.insertHtml(f'<span style="color: #d4d4d4;">{line}</span><br>')

    def apply_log_filter(self):
        """Apply log filter"""
        self.log_text.clear()
        self.last_log_position = 0
        self.refresh_logs()

    def clear_logs(self):
        """Clear log display"""
        self.log_text.clear()

    # ==================== SETTINGS TAB ====================
    def create_settings_tab(self) -> QWidget:
        """Create the settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Tickers
        tickers_group = QGroupBox("Trading Tickers")
        tickers_layout = QVBoxLayout()
        tickers_layout.addWidget(QLabel("Tickers (one per line):"))
        self.tickers_text = QTextEdit()
        self.tickers_text.setMaximumHeight(120)
        tickers_layout.addWidget(self.tickers_text)
        tickers_group.setLayout(tickers_layout)
        layout.addWidget(tickers_group)

        # Risk Parameters
        risk_group = QGroupBox("Risk Parameters")
        risk_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Risk Per Trade (%):"))
        self.risk_per_trade = QDoubleSpinBox()
        self.risk_per_trade.setRange(0.01, 10.0)
        self.risk_per_trade.setSingleStep(0.1)
        self.risk_per_trade.setDecimals(2)
        row1.addWidget(self.risk_per_trade)
        row1.addStretch()
        risk_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Max Position (%):"))
        self.max_position = QDoubleSpinBox()
        self.max_position.setRange(1.0, 50.0)
        self.max_position.setSingleStep(1.0)
        self.max_position.setDecimals(1)
        row2.addWidget(self.max_position)
        row2.addStretch()
        risk_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Stop Loss (%):"))
        self.stop_loss = QDoubleSpinBox()
        self.stop_loss.setRange(0.01, 0.5)
        self.stop_loss.setSingleStep(0.01)
        self.stop_loss.setDecimals(2)
        row3.addWidget(self.stop_loss)
        row3.addStretch()
        risk_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Take Profit (%):"))
        self.take_profit = QDoubleSpinBox()
        self.take_profit.setRange(0.01, 1.0)
        self.take_profit.setSingleStep(0.01)
        self.take_profit.setDecimals(2)
        row4.addWidget(self.take_profit)
        row4.addStretch()
        risk_layout.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Buying Power (%):"))
        self.buying_power = QDoubleSpinBox()
        self.buying_power.setRange(10.0, 100.0)
        self.buying_power.setSingleStep(5.0)
        self.buying_power.setDecimals(0)
        row5.addWidget(self.buying_power)
        row5.addStretch()
        risk_layout.addLayout(row5)

        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)

        # Crypto
        crypto_group = QGroupBox("Cryptocurrency Trading")
        crypto_layout = QVBoxLayout()

        self.crypto_enabled = QCheckBox("Enable Crypto Trading")
        self.crypto_enabled.stateChanged.connect(self.toggle_crypto_settings)
        crypto_layout.addWidget(self.crypto_enabled)

        crypto_layout.addWidget(QLabel("Crypto Tickers (one per line):"))
        self.crypto_tickers_text = QTextEdit()
        self.crypto_tickers_text.setMaximumHeight(80)
        self.crypto_tickers_text.setPlaceholderText("BTC/USD\nETH/USD")
        crypto_layout.addWidget(self.crypto_tickers_text)

        crypto_params = QHBoxLayout()
        crypto_params.addWidget(QLabel("Max Position %:"))
        self.crypto_max_position = QDoubleSpinBox()
        self.crypto_max_position.setRange(0.5, 20.0)
        self.crypto_max_position.setValue(2.5)
        crypto_params.addWidget(self.crypto_max_position)

        crypto_params.addWidget(QLabel("Stop Loss %:"))
        self.crypto_stop_loss = QDoubleSpinBox()
        self.crypto_stop_loss.setRange(1.0, 50.0)
        self.crypto_stop_loss.setValue(15)
        crypto_params.addWidget(self.crypto_stop_loss)
        crypto_params.addStretch()
        crypto_layout.addLayout(crypto_params)

        crypto_group.setLayout(crypto_layout)
        layout.addWidget(crypto_group)

        # Buttons
        buttons = QHBoxLayout()
        save_btn = QPushButton("💾 Save Settings")
        save_btn.clicked.connect(self.save_settings_config)
        buttons.addWidget(save_btn)

        reload_btn = QPushButton("🔄 Reload")
        reload_btn.clicked.connect(self.load_settings_config)
        buttons.addWidget(reload_btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        info_label = QLabel("⚠️ Bot must be restarted for changes to take effect")
        info_label.setStyleSheet("color: orange;")
        layout.addWidget(info_label)

        layout.addStretch()
        return widget

    def toggle_crypto_settings(self, state):
        """Enable/disable crypto settings"""
        enabled = state == Qt.CheckState.Checked.value
        self.crypto_tickers_text.setEnabled(enabled)
        self.crypto_max_position.setEnabled(enabled)
        self.crypto_stop_loss.setEnabled(enabled)

    def load_settings_config(self):
        """Load config"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            self.tickers_text.setPlainText('\n'.join(self.config.get('tickers', [])))
            self.risk_per_trade.setValue(self.config.get('risk_per_trade_pct', 0.2))
            self.max_position.setValue(self.config.get('max_position_pct', 5.0))
            self.stop_loss.setValue(self.config.get('stop_loss_pct', 0.05))
            self.take_profit.setValue(self.config.get('take_profit_pct', 0.1))
            self.buying_power.setValue(self.config.get('buying_power_pct', 50))

            crypto = self.config.get('crypto', {})
            self.crypto_enabled.setChecked(crypto.get('enabled', False))
            self.crypto_tickers_text.setPlainText('\n'.join(crypto.get('tickers', [])))
            self.crypto_max_position.setValue(crypto.get('max_position_pct', 2.5))
            self.crypto_stop_loss.setValue(crypto.get('stop_loss_pct', 15) * 100 if crypto.get('stop_loss_pct', 15) < 1 else crypto.get('stop_loss_pct', 15))

            self.toggle_crypto_settings(Qt.CheckState.Checked.value if crypto.get('enabled', False) else Qt.CheckState.Unchecked.value)
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    def save_settings_config(self):
        """Save config"""
        try:
            tickers = [t.strip() for t in self.tickers_text.toPlainText().split('\n') if t.strip()]
            self.config['tickers'] = tickers
            self.config['risk_per_trade_pct'] = self.risk_per_trade.value()
            self.config['max_position_pct'] = self.max_position.value()
            self.config['stop_loss_pct'] = self.stop_loss.value()
            self.config['take_profit_pct'] = self.take_profit.value()
            self.config['buying_power_pct'] = int(self.buying_power.value())

            crypto_tickers = [t.strip() for t in self.crypto_tickers_text.toPlainText().split('\n') if t.strip()]
            if 'crypto' not in self.config:
                self.config['crypto'] = {}
            self.config['crypto']['enabled'] = self.crypto_enabled.isChecked()
            self.config['crypto']['tickers'] = crypto_tickers
            self.config['crypto']['max_position_pct'] = self.crypto_max_position.value()
            self.config['crypto']['stop_loss_pct'] = self.crypto_stop_loss.value() / 100

            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)

            QMessageBox.information(self, "Success", "Settings saved! Restart bot for changes to take effect.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    # ==================== BACKTEST TAB ====================
    def create_backtest_tab(self) -> QWidget:
        """Create backtest tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Header
        header = QHBoxLayout()
        self.backtest_status_label = QLabel("Status: —")
        header.addWidget(self.backtest_status_label)
        header.addStretch()

        run_btn = QPushButton("▶ Run Backtest")
        run_btn.clicked.connect(self.run_backtest)
        header.addWidget(run_btn)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.load_backtest_artifacts)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        # Sub-tabs for results
        self.backtest_tabs = QTabWidget()
        layout.addWidget(self.backtest_tabs)

        # Report
        self.bt_report_text = QTextEdit()
        self.bt_report_text.setReadOnly(True)
        self.backtest_tabs.addTab(self.bt_report_text, "Report")

        # Chart
        self.bt_chart_label = QLabel("No chart yet.")
        self.bt_chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bt_scroll = QScrollArea()
        bt_scroll.setWidgetResizable(True)
        bt_scroll.setWidget(self.bt_chart_label)
        self.backtest_tabs.addTab(bt_scroll, "Chart")

        # Results table
        self.bt_results_table = QTableWidget()
        self.backtest_tabs.addTab(self.bt_results_table, "Results")

        return widget

    def run_backtest(self):
        """Run backtest"""
        resp = self.ipc_client.send_command({"command": "run_backtest"}, timeout=5.0)
        if resp.get("success"):
            QMessageBox.information(self, "Backtest", "Backtest started.")
        else:
            QMessageBox.critical(self, "Backtest", f"Failed: {resp.get('message') or resp.get('error')}")

    def refresh_backtest_status(self):
        """Refresh backtest status"""
        resp = self.ipc_client.send_command({"command": "get_backtest_status"}, timeout=5.0)
        if resp.get("error"):
            return
        running = resp.get("running", False)
        phase = resp.get("phase", "—")
        progress = resp.get("progress", "—")
        if running:
            self.backtest_status_label.setText(f"Status: Running - {phase}: {progress}")
        else:
            last_run = resp.get("last_run", "—")
            self.backtest_status_label.setText(f"Status: Idle (Last: {last_run})")

    def load_backtest_artifacts(self):
        """Load backtest artifacts"""
        if self.report_path.exists():
            self.bt_report_text.setPlainText(self.report_path.read_text())
        else:
            self.bt_report_text.setPlainText("No report yet. Run backtest first.")

        if self.bt_png_path.exists():
            pixmap = QPixmap(str(self.bt_png_path))
            if not pixmap.isNull():
                self.bt_chart_label.setPixmap(pixmap)
        else:
            self.bt_chart_label.setText("No chart yet.")

        if self.bt_csv_path.exists():
            try:
                df = pd.read_csv(self.bt_csv_path)
                self.bt_results_table.setRowCount(len(df))
                self.bt_results_table.setColumnCount(len(df.columns))
                self.bt_results_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
                for r in range(len(df)):
                    for c, col in enumerate(df.columns):
                        self.bt_results_table.setItem(r, c, QTableWidgetItem(str(df.iloc[r][col])))
                self.bt_results_table.resizeColumnsToContents()
            except Exception:
                pass

    # ==================== CHAT TAB ====================
    def create_chat_tab(self) -> QWidget:
        """Create chat tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Header
        top = QHBoxLayout()
        self.chat_status_label = QLabel("Ready")
        top.addWidget(self.chat_status_label)
        top.addStretch()
        self.include_context = QCheckBox("Include account/positions/signals")
        self.include_context.setChecked(True)
        top.addWidget(self.include_context)
        layout.addLayout(top)

        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(self.chat_history)

        # Input
        bottom = QHBoxLayout()
        self.chat_input = QTextEdit()
        self.chat_input.setFixedHeight(80)
        bottom.addWidget(self.chat_input, stretch=1)

        right = QVBoxLayout()
        self.chat_send_btn = QPushButton("💬 Send to Dexter")
        self.chat_send_btn.clicked.connect(self.send_chat)
        right.addWidget(self.chat_send_btn)

        self.claude_analyze_btn = QPushButton("🧠 Ask Claude")
        self.claude_analyze_btn.clicked.connect(self.ask_claude_analysis)
        self.claude_analyze_btn.setToolTip("Have Claude analyze Dexter's last response")
        self.claude_analyze_btn.setEnabled(False)
        right.addWidget(self.claude_analyze_btn)

        right.addWidget(QLabel(""))  # Spacer

        self.dexter_tickers_btn = QPushButton("📋 Use Dexter Tickers")
        self.dexter_tickers_btn.clicked.connect(self.use_dexter_tickers)
        self.dexter_tickers_btn.setToolTip("Get ticker recommendations from Dexter and update settings")
        right.addWidget(self.dexter_tickers_btn)

        self.claude_tickers_btn = QPushButton("🤖 Use Claude Tickers")
        self.claude_tickers_btn.clicked.connect(self.use_claude_tickers)
        self.claude_tickers_btn.setToolTip("Get fresh stock & crypto recommendations from Claude and update settings")
        right.addWidget(self.claude_tickers_btn)

        self.dexter_bias_btn = QPushButton("📊 Generate Bias")
        self.dexter_bias_btn.clicked.connect(self.generate_dexter_bias)
        right.addWidget(self.dexter_bias_btn)

        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self.clear_chat)
        right.addWidget(clear_btn)
        right.addStretch()
        bottom.addLayout(right)
        layout.addLayout(bottom)

        return widget

    def clear_chat(self):
        """Clear chat and reset state"""
        self.chat_history.clear()
        self.last_dexter_response = ""
        self.claude_analyze_btn.setEnabled(False)

    def send_chat(self):
        """Send chat message"""
        query = self.chat_input.toPlainText().strip()
        if not query:
            return
        self.chat_input.clear()
        self.chat_history.append(f"<b>You:</b> {query}")
        self.chat_status_label.setText("Asking Dexter...")
        self.chat_send_btn.setEnabled(False)

        include_ctx = self.include_context.isChecked()

        def worker():
            resp = self.ipc_client.send_command(
                {"command": "dexter_chat", "query": query, "include_context": include_ctx},
                timeout=180.0,
            )
            self.chat_signals.finished.emit(resp)

        threading.Thread(target=worker, daemon=True).start()

    def _on_chat_response(self, resp: dict):
        """Handle chat response"""
        self.chat_send_btn.setEnabled(True)
        if resp.get("error"):
            self.chat_status_label.setText("Error")
            self.chat_history.append(f"<b>Error:</b> {resp.get('error')}")
        else:
            self.chat_status_label.setText("Ready")
            answer = resp.get('answer', '')
            self.chat_history.append(f"<b>Dexter:</b> {answer}")
            # Store for Claude analysis
            self.last_dexter_response = answer
            self.claude_analyze_btn.setEnabled(True)

    def ask_claude_analysis(self):
        """Ask Claude to analyze Dexter's last response"""
        if not self.last_dexter_response:
            QMessageBox.information(self, "Claude", "No Dexter response to analyze yet.")
            return

        self.chat_history.append("<b>You:</b> [Asking Claude to analyze Dexter's response...]")
        self.chat_status_label.setText("Asking Claude...")
        self.claude_analyze_btn.setEnabled(False)
        self.chat_send_btn.setEnabled(False)

        def worker():
            try:
                import anthropic
                import os

                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    # Try to load from config
                    try:
                        with open(self.config_path, 'r') as f:
                            cfg = yaml.safe_load(f)
                        api_key = cfg.get('claude', {}).get('api_key') or cfg.get('anthropic', {}).get('api_key')
                    except Exception:
                        pass

                if not api_key:
                    self.chat_signals.claude_finished.emit({"error": "No Claude API key found. Set ANTHROPIC_API_KEY or add to config.yaml"})
                    return

                client = anthropic.Anthropic(api_key=api_key)

                prompt = f"""You are a trading analyst. Analyze the following research/analysis from an AI assistant named Dexter and provide your assessment:

DEXTER'S RESPONSE:
{self.last_dexter_response}

Please provide:
1. A brief summary of Dexter's key points
2. Your assessment of the analysis quality and any potential blind spots
3. Any additional considerations or risks not mentioned
4. Your overall recommendation (agree/disagree/partially agree with Dexter's conclusions)

Be concise but thorough."""

                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = message.content[0].text
                self.chat_signals.claude_finished.emit({"answer": response_text})

            except Exception as e:
                self.chat_signals.claude_finished.emit({"error": str(e)})

        threading.Thread(target=worker, daemon=True).start()

    def use_dexter_tickers(self):
        """Update tickers via Dexter and sync to settings"""
        # First, trigger Dexter to update tickers_auto.json
        resp = self.ipc_client.send_command({"command": "dexter_update_tickers"}, timeout=5.0)
        if resp.get("error"):
            QMessageBox.critical(self, "Dexter", resp.get("error"))
            return
        elif resp.get("success") is False:
            QMessageBox.information(self, "Dexter", resp.get("message", "Already running."))
            return

        self.chat_history.append("<b>System:</b> Dexter ticker update started. Waiting for results...")
        self.chat_status_label.setText("Updating tickers...")
        self.dexter_tickers_btn.setEnabled(False)

        # Poll for completion and then update settings
        def wait_and_update():
            import time
            import json

            # Wait for tickers_auto.json to be updated (max 60 seconds)
            tickers_file = Path("tickers_auto.json")
            start_time = time.time()
            original_mtime = tickers_file.stat().st_mtime if tickers_file.exists() else 0

            while time.time() - start_time < 60:
                # Check if Dexter update is still running
                status = self.ipc_client.send_command({"command": "get_dexter_status"}, timeout=5.0)
                if not status.get("tickers_update_running", False):
                    break
                time.sleep(2)

            # Check if file was updated
            if tickers_file.exists():
                new_mtime = tickers_file.stat().st_mtime
                if new_mtime > original_mtime or original_mtime == 0:
                    try:
                        with open(tickers_file, 'r') as f:
                            data = json.load(f)
                        new_tickers = data.get("tickers", [])

                        if new_tickers:
                            # Update config.yaml with new tickers
                            with open(self.config_path, 'r') as f:
                                config = yaml.safe_load(f)

                            old_tickers = config.get('tickers', [])
                            config['tickers'] = new_tickers

                            with open(self.config_path, 'w') as f:
                                yaml.dump(config, f, default_flow_style=False)

                            # Signal success
                            self.chat_signals.claude_finished.emit({
                                "answer": f"Tickers updated successfully!\n\nOld tickers: {', '.join(old_tickers)}\n\nNew tickers: {', '.join(new_tickers)}\n\n⚠️ Restart bot for changes to take effect.",
                                "_tickers_updated": True,
                                "_new_tickers": new_tickers
                            })
                            return
                    except Exception as e:
                        self.chat_signals.claude_finished.emit({"error": f"Failed to update tickers: {e}"})
                        return

            self.chat_signals.claude_finished.emit({"error": "Ticker update timed out or no new tickers found."})

        threading.Thread(target=wait_and_update, daemon=True).start()

    def use_claude_tickers(self):
        """Ask Claude for fresh stock and crypto recommendations and update settings"""
        self.chat_history.append("<b>You:</b> [Asking Claude for stock & crypto recommendations...]")
        self.chat_status_label.setText("Getting Claude recommendations...")
        self.claude_tickers_btn.setEnabled(False)
        self.dexter_tickers_btn.setEnabled(False)
        self.chat_send_btn.setEnabled(False)

        def worker():
            try:
                import anthropic
                import os
                import json
                import re

                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    try:
                        with open(self.config_path, 'r') as f:
                            cfg = yaml.safe_load(f)
                        api_key = cfg.get('claude', {}).get('api_key') or cfg.get('anthropic', {}).get('api_key')
                    except Exception:
                        pass

                if not api_key:
                    self.chat_signals.claude_finished.emit({"error": "No Claude API key found. Set ANTHROPIC_API_KEY or add to config.yaml"})
                    return

                client = anthropic.Anthropic(api_key=api_key)

                # Get current date for context
                from datetime import datetime
                today = datetime.now().strftime("%B %d, %Y")

                prompt = f"""You are a trading analyst. Today is {today}.

Please recommend stocks and cryptocurrencies for day/swing trading based on current market conditions.

Requirements:
1. Recommend 5-10 stock tickers (US stocks only, use standard symbols like AAPL, TSLA, NVDA)
2. Recommend 2-4 crypto pairs (use format like BTC/USD, ETH/USD)
3. Focus on liquid, actively traded assets with good volatility for trading
4. Consider current market trends, momentum, and technical setups

IMPORTANT: Return your response in this EXACT JSON format at the end of your message:
```json
{{
  "stocks": ["TICKER1", "TICKER2", "TICKER3"],
  "crypto": ["BTC/USD", "ETH/USD"],
  "reasoning": "Brief explanation of your picks"
}}
```

First provide your analysis, then end with the JSON block."""

                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = message.content[0].text

                # Parse the JSON from Claude's response
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if not json_match:
                    # Try without code blocks
                    json_match = re.search(r'\{[^{}]*"stocks"[^{}]*"crypto"[^{}]*\}', response_text, re.DOTALL)

                if json_match:
                    try:
                        data = json.loads(json_match.group(1) if '```' in response_text else json_match.group(0))
                        stocks = data.get("stocks", [])
                        crypto = data.get("crypto", [])
                        reasoning = data.get("reasoning", "")

                        if stocks or crypto:
                            # Update config.yaml with new tickers
                            with open(self.config_path, 'r') as f:
                                config = yaml.safe_load(f)

                            old_tickers = config.get('tickers', [])
                            old_crypto = config.get('crypto', {}).get('tickers', [])

                            # Update stock tickers
                            if stocks:
                                config['tickers'] = stocks

                            # Update crypto tickers
                            if crypto:
                                if 'crypto' not in config:
                                    config['crypto'] = {'enabled': True}
                                config['crypto']['tickers'] = crypto
                                config['crypto']['enabled'] = True

                            with open(self.config_path, 'w') as f:
                                yaml.dump(config, f, default_flow_style=False)

                            result_msg = f"""**Claude's Recommendations:**

{reasoning}

**Tickers Updated:**
• Old stocks: {', '.join(old_tickers) if old_tickers else 'None'}
• New stocks: {', '.join(stocks) if stocks else 'No change'}

• Old crypto: {', '.join(old_crypto) if old_crypto else 'None'}
• New crypto: {', '.join(crypto) if crypto else 'No change'}

⚠️ **Restart bot for changes to take effect.**"""

                            self.chat_signals.claude_finished.emit({
                                "answer": result_msg,
                                "_tickers_updated": True,
                                "_new_tickers": stocks,
                                "_new_crypto": crypto
                            })
                            return

                    except json.JSONDecodeError as e:
                        pass

                # If we couldn't parse JSON, just show the response
                self.chat_signals.claude_finished.emit({
                    "answer": f"Claude's response (couldn't auto-update tickers):\n\n{response_text}",
                    "_claude_recommendation": True
                })

            except Exception as e:
                self.chat_signals.claude_finished.emit({"error": str(e)})

        threading.Thread(target=worker, daemon=True).start()

    def _on_claude_response(self, resp: dict):
        """Handle Claude analysis response or ticker update response"""
        self.chat_send_btn.setEnabled(True)
        self.claude_analyze_btn.setEnabled(bool(self.last_dexter_response))
        self.dexter_tickers_btn.setEnabled(True)
        self.claude_tickers_btn.setEnabled(True)
        self.chat_status_label.setText("Ready")

        if resp.get("error"):
            self.chat_history.append(f"<b>Error:</b> {resp.get('error')}")
        else:
            answer = resp.get('answer', '')
            # Check if this was a ticker update
            if resp.get("_tickers_updated"):
                self.chat_history.append(f"<b>System:</b> {answer}")
                # Reload settings tab to show new tickers
                self.load_settings_config()
            elif resp.get("_claude_recommendation"):
                self.chat_history.append(f"<b>Claude:</b> {answer}")
            else:
                self.chat_history.append(f"<b>Claude Analysis:</b> {answer}")

    def generate_dexter_bias(self):
        """Generate Dexter bias"""
        resp = self.ipc_client.send_command({"command": "dexter_generate_bias"}, timeout=5.0)
        if resp.get("error"):
            QMessageBox.critical(self, "Dexter", resp.get("error"))
        elif resp.get("success"):
            QMessageBox.information(self, "Dexter", "Bias generation started.")
        else:
            QMessageBox.information(self, "Dexter", resp.get("message", "Already running."))

    def closeEvent(self, event):
        """Handle close - hide instead"""
        event.ignore()
        self.hide()
