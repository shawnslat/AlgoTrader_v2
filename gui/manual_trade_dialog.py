"""
Manual Trade Dialog - Execute manual trades
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QSpinBox, QRadioButton, QPushButton,
    QButtonGroup, QMessageBox
)
from PyQt6.QtCore import Qt
from ipc_protocol import IPCClient
import logging

logger = logging.getLogger(__name__)


class ManualTradeDialog(QDialog):
    """Dialog for executing manual trades"""

    def __init__(self, ipc_client: IPCClient, parent=None):
        super().__init__(parent)
        self.ipc_client = ipc_client
        self.setup_ui()

    def setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle("Manual Trade")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Ticker input
        ticker_layout = QHBoxLayout()
        ticker_layout.addWidget(QLabel("Ticker:"))
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("e.g., AAPL")
        ticker_layout.addWidget(self.ticker_input)
        layout.addLayout(ticker_layout)

        # Action selection
        action_layout = QHBoxLayout()
        action_layout.addWidget(QLabel("Action:"))

        self.action_group = QButtonGroup(self)
        self.buy_radio = QRadioButton("Buy")
        self.sell_radio = QRadioButton("Sell")
        self.buy_radio.setChecked(True)

        self.action_group.addButton(self.buy_radio)
        self.action_group.addButton(self.sell_radio)

        action_layout.addWidget(self.buy_radio)
        action_layout.addWidget(self.sell_radio)
        action_layout.addStretch()
        layout.addLayout(action_layout)

        # Quantity input
        qty_layout = QHBoxLayout()
        qty_layout.addWidget(QLabel("Quantity:"))
        self.qty_input = QSpinBox()
        self.qty_input.setRange(1, 10000)
        self.qty_input.setValue(10)
        qty_layout.addWidget(self.qty_input)
        qty_layout.addStretch()
        layout.addLayout(qty_layout)

        # Warning label
        warning_label = QLabel("⚠️ This will execute a real market order!")
        warning_label.setStyleSheet("color: orange; font-weight: bold;")
        layout.addWidget(warning_label)

        # Buttons
        buttons_layout = QHBoxLayout()

        execute_btn = QPushButton("Execute Trade")
        execute_btn.clicked.connect(self.execute_trade)
        execute_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        buttons_layout.addWidget(execute_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)

        layout.addLayout(buttons_layout)

    def execute_trade(self):
        """Execute the manual trade"""
        ticker = self.ticker_input.text().strip().upper()
        action = "buy" if self.buy_radio.isChecked() else "sell"
        quantity = self.qty_input.value()

        # Validation
        if not ticker:
            QMessageBox.warning(self, "Invalid Input", "Please enter a ticker symbol")
            return

        # Confirmation
        confirm = QMessageBox.question(
            self,
            "Confirm Trade",
            f"Execute {action.upper()} {quantity} shares of {ticker}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if confirm != QMessageBox.StandardButton.Yes:
            return

        # Execute via IPC
        response = self.ipc_client.send_command({
            "command": "manual_trade",
            "ticker": ticker,
            "action": action,
            "quantity": quantity
        })

        if response.get("error"):
            QMessageBox.critical(
                self,
                "Trade Failed",
                f"Failed to execute trade:\n{response['error']}"
            )
            logger.error(f"Manual trade failed: {response['error']}")
        elif response.get("success"):
            QMessageBox.information(
                self,
                "Trade Executed",
                f"Successfully executed:\n{response.get('message', 'Trade complete')}\n\nOrder ID: {response.get('order_id', 'N/A')}"
            )
            logger.info(f"Manual trade executed: {action} {quantity} {ticker}")
            self.accept()
        else:
            QMessageBox.warning(
                self,
                "Trade Status Unknown",
                "Trade status unclear. Check your positions."
            )
