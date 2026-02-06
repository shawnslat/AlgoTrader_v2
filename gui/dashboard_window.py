"""
Dashboard Window - Main portfolio and trading view
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTableWidget, QTableWidgetItem, QPushButton,
    QGroupBox, QMessageBox, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from ipc_protocol import IPCClient
import logging

logger = logging.getLogger(__name__)


def is_crypto_ticker(ticker: str) -> bool:
    """Check if ticker is a cryptocurrency"""
    return '/' in ticker or ticker.endswith('USD') and len(ticker) <= 7


class DashboardWindow(QMainWindow):
    """Main dashboard window showing portfolio, positions, and signals"""

    def __init__(self, ipc_client: IPCClient):
        super().__init__()
        self.ipc_client = ipc_client

        # Data
        self.account_data = {}
        self.positions_data = []
        self.signals_data = []

        # Filter state
        self.position_filter = "all"  # "all", "stocks", "crypto"

        self.setup_ui()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds

        # Initial data load
        self.refresh_data()

    def setup_ui(self):
        """Setup the dashboard UI"""
        self.setWindowTitle("Trading Bot - Dashboard")
        self.setGeometry(100, 100, 1000, 700)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Account Summary Section
        account_group = self.create_account_summary()
        layout.addWidget(account_group)

        # Positions Section
        positions_group = self.create_positions_table()
        layout.addWidget(positions_group)

        # Signals Section
        signals_group = self.create_signals_table()
        layout.addWidget(signals_group)

        # Buttons Section
        buttons_layout = QHBoxLayout()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_data)
        buttons_layout.addWidget(refresh_btn)

        refresh_signals_btn = QPushButton("🧠 Refresh Signals")
        refresh_signals_btn.clicked.connect(self.refresh_signals_now)
        buttons_layout.addWidget(refresh_signals_btn)

        manual_trade_btn = QPushButton("📊 Manual Trade")
        manual_trade_btn.clicked.connect(self.open_manual_trade)
        buttons_layout.addWidget(manual_trade_btn)

        buttons_layout.addStretch()

        layout.addLayout(buttons_layout)

    def create_account_summary(self) -> QGroupBox:
        """Create account summary section"""
        group = QGroupBox("Account Summary")
        layout = QHBoxLayout()

        # Labels for account metrics
        self.cash_label = QLabel("Cash: $0.00")
        self.cash_label.setFont(QFont("Arial", 14))

        self.buying_power_label = QLabel("Buying Power: $0.00")
        self.buying_power_label.setFont(QFont("Arial", 14))

        self.portfolio_value_label = QLabel("Portfolio Value: $0.00")
        self.portfolio_value_label.setFont(QFont("Arial", 14))

        self.equity_label = QLabel("Equity: $0.00")
        self.equity_label.setFont(QFont("Arial", 14))

        layout.addWidget(self.cash_label)
        layout.addWidget(self.buying_power_label)
        layout.addWidget(self.portfolio_value_label)
        layout.addWidget(self.equity_label)
        layout.addStretch()

        group.setLayout(layout)
        return group

    def create_positions_table(self) -> QGroupBox:
        """Create positions table"""
        group = QGroupBox("Current Positions")
        layout = QVBoxLayout()

        # Filter buttons
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
        layout.addLayout(filter_layout)

        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(8)
        self.positions_table.setHorizontalHeaderLabels([
            "Type", "Ticker", "Qty", "Entry Price", "Current Price",
            "Market Value", "P&L", "P&L %"
        ])
        self.positions_table.setAlternatingRowColors(True)

        layout.addWidget(self.positions_table)
        group.setLayout(layout)
        return group

    def create_signals_table(self) -> QGroupBox:
        """Create signals table"""
        group = QGroupBox("Today's Trading Signals")
        layout = QVBoxLayout()

        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(4)
        self.signals_table.setHorizontalHeaderLabels([
            "Ticker", "Signal", "Confidence", "Time"
        ])
        self.signals_table.setAlternatingRowColors(True)

        layout.addWidget(self.signals_table)
        group.setLayout(layout)
        return group

    def refresh_data(self):
        """Refresh all dashboard data"""
        self.refresh_account()
        self.refresh_positions()
        self.refresh_signals()

    def refresh_account(self):
        """Refresh account summary"""
        response = self.ipc_client.send_command({"command": "get_account"})

        if response.get("error"):
            logger.error(f"Error fetching account: {response['error']}")
            return

        self.account_data = response

        # Update labels
        cash = response.get("cash", 0)
        buying_power = response.get("buying_power", 0)
        portfolio_value = response.get("portfolio_value", 0)
        equity = response.get("equity", 0)

        self.cash_label.setText(f"Cash: ${cash:,.2f}")
        self.buying_power_label.setText(f"Buying Power: ${buying_power:,.2f}")
        self.portfolio_value_label.setText(f"Portfolio Value: ${portfolio_value:,.2f}")
        self.equity_label.setText(f"Equity: ${equity:,.2f}")

    def set_position_filter(self, filter_type: str):
        """Set position filter and refresh display"""
        self.position_filter = filter_type
        self.update_positions_display()

    def refresh_positions(self):
        """Refresh positions table"""
        response = self.ipc_client.send_command({"command": "get_positions"})

        if response.get("error"):
            logger.error(f"Error fetching positions: {response['error']}")
            return

        positions = response.get("positions", [])
        self.positions_data = positions

        # Update display with current filter
        self.update_positions_display()

    def update_positions_display(self):
        """Update positions table based on current filter"""
        # Filter positions based on current filter
        filtered_positions = []
        for pos in self.positions_data:
            ticker = pos.get("ticker", "")
            is_crypto = is_crypto_ticker(ticker)

            if self.position_filter == "all":
                filtered_positions.append(pos)
            elif self.position_filter == "stocks" and not is_crypto:
                filtered_positions.append(pos)
            elif self.position_filter == "crypto" and is_crypto:
                filtered_positions.append(pos)

        # Update table
        self.positions_table.setRowCount(len(filtered_positions))

        for row, pos in enumerate(filtered_positions):
            ticker = pos.get("ticker", "")
            qty = pos.get("qty", pos.get("quantity", 0))
            entry_price = pos.get("avg_entry_price", pos.get("average_price", 0))
            current_price = pos.get("current_price", 0)
            market_value = pos.get("market_value", 0)
            unrealized_pl = pos.get("unrealized_pl", 0)
            unrealized_plpc = pos.get("unrealized_plpc", 0)

            # Asset type icon
            is_crypto = is_crypto_ticker(ticker)
            type_icon = "🪙" if is_crypto else "📈"
            self.positions_table.setItem(row, 0, QTableWidgetItem(type_icon))

            self.positions_table.setItem(row, 1, QTableWidgetItem(ticker))

            # Format quantity based on asset type
            if is_crypto:
                # Show more decimals for crypto
                qty_str = f"{qty:.8f}".rstrip('0').rstrip('.')
            else:
                qty_str = str(int(qty)) if qty == int(qty) else f"{qty:.2f}"

            self.positions_table.setItem(row, 2, QTableWidgetItem(qty_str))
            self.positions_table.setItem(row, 3, QTableWidgetItem(f"${entry_price:.2f}"))
            self.positions_table.setItem(row, 4, QTableWidgetItem(f"${current_price:.2f}"))
            self.positions_table.setItem(row, 5, QTableWidgetItem(f"${market_value:.2f}"))

            # Color P&L based on positive/negative
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
        """Refresh signals table"""
        response = self.ipc_client.send_command({"command": "get_signals"})

        if response.get("error"):
            logger.error(f"Error fetching signals: {response['error']}")
            return

        signals = response.get("signals", [])
        self.signals_data = signals

        # Update table
        self.signals_table.setRowCount(len(signals))

        for row, signal in enumerate(signals):
            ticker = signal.get("ticker", "")
            action = signal.get("action", "")
            confidence = signal.get("confidence", 0)
            timestamp = signal.get("timestamp", "")

            self.signals_table.setItem(row, 0, QTableWidgetItem(ticker))

            # Color signal based on action
            signal_item = QTableWidgetItem(action.upper())
            if action.lower() == "buy":
                signal_item.setForeground(Qt.GlobalColor.darkGreen)
            elif action.lower() == "sell":
                signal_item.setForeground(Qt.GlobalColor.red)

            self.signals_table.setItem(row, 1, signal_item)
            self.signals_table.setItem(row, 2, QTableWidgetItem(f"{confidence:.2f}"))
            self.signals_table.setItem(row, 3, QTableWidgetItem(timestamp))

        self.signals_table.resizeColumnsToContents()

    def refresh_signals_now(self):
        """Force a signals refresh from the service."""
        response = self.ipc_client.send_command({"command": "refresh_signals"}, timeout=2.0)
        if response.get("error"):
            QMessageBox.warning(self, "Signals", f"Failed to start refresh: {response.get('error')}")
            return
        if response.get("success") is False:
            QMessageBox.information(self, "Signals", response.get("message", "Refresh already running."))
            return
        QMessageBox.information(self, "Signals", "Refresh started. Signals will update when ready.")

    def focus_positions(self):
        try:
            self.positions_table.setFocus()
        except Exception:
            pass

    def focus_signals(self):
        try:
            self.signals_table.setFocus()
        except Exception:
            pass

        self.signals_table.resizeColumnsToContents()

    def open_manual_trade(self):
        """Open manual trade dialog"""
        from gui.manual_trade_dialog import ManualTradeDialog

        dialog = ManualTradeDialog(self.ipc_client, self)
        dialog.exec()

        # Refresh data after trade
        self.refresh_data()

    def closeEvent(self, event):
        """Handle window close"""
        # Just hide the window instead of closing
        event.ignore()
        self.hide()
