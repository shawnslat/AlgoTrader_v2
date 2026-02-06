"""
Settings Window - Edit bot configuration
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QGroupBox, QMessageBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QSpinBox
)
from PyQt6.QtCore import Qt
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SettingsWindow(QMainWindow):
    """Window for editing bot settings"""

    def __init__(self):
        super().__init__()
        self.config_path = Path("config.yaml")
        self.config = {}

        self.setup_ui()
        self.load_config()

    def setup_ui(self):
        """Setup the settings window UI"""
        self.setWindowTitle("Trading Bot - Settings")
        self.setGeometry(100, 100, 800, 800)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Tickers section
        tickers_group = QGroupBox("Trading Tickers")
        tickers_layout = QVBoxLayout()

        tickers_layout.addWidget(QLabel("Tickers (one per line):"))
        self.tickers_text = QTextEdit()
        self.tickers_text.setMaximumHeight(150)
        tickers_layout.addWidget(self.tickers_text)

        tickers_group.setLayout(tickers_layout)
        layout.addWidget(tickers_group)

        # Risk parameters section
        risk_group = QGroupBox("Risk Parameters")
        risk_layout = QVBoxLayout()

        # Risk per trade
        risk_trade_layout = QHBoxLayout()
        risk_trade_layout.addWidget(QLabel("Risk Per Trade (%):"))
        self.risk_per_trade = QDoubleSpinBox()
        self.risk_per_trade.setRange(0.01, 10.0)
        self.risk_per_trade.setSingleStep(0.1)
        self.risk_per_trade.setDecimals(2)
        risk_trade_layout.addWidget(self.risk_per_trade)
        risk_trade_layout.addStretch()
        risk_layout.addLayout(risk_trade_layout)

        # Max position
        max_pos_layout = QHBoxLayout()
        max_pos_layout.addWidget(QLabel("Max Position (%):"))
        self.max_position = QDoubleSpinBox()
        self.max_position.setRange(1.0, 50.0)
        self.max_position.setSingleStep(1.0)
        self.max_position.setDecimals(1)
        max_pos_layout.addWidget(self.max_position)
        max_pos_layout.addStretch()
        risk_layout.addLayout(max_pos_layout)

        # Stop loss
        stop_loss_layout = QHBoxLayout()
        stop_loss_layout.addWidget(QLabel("Stop Loss (%):"))
        self.stop_loss = QDoubleSpinBox()
        self.stop_loss.setRange(0.01, 0.5)
        self.stop_loss.setSingleStep(0.01)
        self.stop_loss.setDecimals(2)
        stop_loss_layout.addWidget(self.stop_loss)
        stop_loss_layout.addStretch()
        risk_layout.addLayout(stop_loss_layout)

        # Take profit
        take_profit_layout = QHBoxLayout()
        take_profit_layout.addWidget(QLabel("Take Profit (%):"))
        self.take_profit = QDoubleSpinBox()
        self.take_profit.setRange(0.01, 1.0)
        self.take_profit.setSingleStep(0.01)
        self.take_profit.setDecimals(2)
        take_profit_layout.addWidget(self.take_profit)
        take_profit_layout.addStretch()
        risk_layout.addLayout(take_profit_layout)

        # Buying power
        buying_power_layout = QHBoxLayout()
        buying_power_layout.addWidget(QLabel("Buying Power Usage (%):"))
        self.buying_power = QDoubleSpinBox()
        self.buying_power.setRange(10.0, 100.0)
        self.buying_power.setSingleStep(5.0)
        self.buying_power.setDecimals(0)
        buying_power_layout.addWidget(self.buying_power)
        buying_power_layout.addStretch()
        risk_layout.addLayout(buying_power_layout)

        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)

        # Crypto Trading section
        crypto_group = QGroupBox("Cryptocurrency Trading")
        crypto_layout = QVBoxLayout()

        # Enable crypto checkbox
        crypto_enable_layout = QHBoxLayout()
        self.crypto_enabled = QCheckBox("Enable Crypto Trading")
        self.crypto_enabled.stateChanged.connect(self.toggle_crypto_settings)
        crypto_enable_layout.addWidget(self.crypto_enabled)
        crypto_enable_layout.addStretch()
        crypto_layout.addLayout(crypto_enable_layout)

        # Crypto tickers
        crypto_layout.addWidget(QLabel("Crypto Tickers (one per line):"))
        self.crypto_tickers_text = QTextEdit()
        self.crypto_tickers_text.setMaximumHeight(100)
        self.crypto_tickers_text.setPlaceholderText("BTC/USD\nETH/USD")
        crypto_layout.addWidget(self.crypto_tickers_text)

        # Crypto risk parameters
        crypto_params_layout = QVBoxLayout()

        # Max position (crypto)
        crypto_max_pos_layout = QHBoxLayout()
        crypto_max_pos_layout.addWidget(QLabel("Max Position (%):"))
        self.crypto_max_position = QDoubleSpinBox()
        self.crypto_max_position.setRange(0.5, 20.0)
        self.crypto_max_position.setSingleStep(0.5)
        self.crypto_max_position.setDecimals(1)
        self.crypto_max_position.setValue(2.5)
        crypto_max_pos_layout.addWidget(self.crypto_max_position)
        crypto_max_pos_layout.addStretch()
        crypto_params_layout.addLayout(crypto_max_pos_layout)

        # Risk multiplier
        crypto_risk_mult_layout = QHBoxLayout()
        crypto_risk_mult_layout.addWidget(QLabel("Risk Multiplier:"))
        self.crypto_risk_multiplier = QDoubleSpinBox()
        self.crypto_risk_multiplier.setRange(0.5, 3.0)
        self.crypto_risk_multiplier.setSingleStep(0.1)
        self.crypto_risk_multiplier.setDecimals(1)
        self.crypto_risk_multiplier.setValue(1.5)
        crypto_risk_mult_layout.addWidget(self.crypto_risk_multiplier)
        crypto_risk_mult_layout.addStretch()
        crypto_params_layout.addLayout(crypto_risk_mult_layout)

        # Stop loss (crypto)
        crypto_stop_loss_layout = QHBoxLayout()
        crypto_stop_loss_layout.addWidget(QLabel("Stop Loss (%):"))
        self.crypto_stop_loss = QDoubleSpinBox()
        self.crypto_stop_loss.setRange(1.0, 50.0)
        self.crypto_stop_loss.setSingleStep(1.0)
        self.crypto_stop_loss.setDecimals(0)
        self.crypto_stop_loss.setValue(15)
        crypto_stop_loss_layout.addWidget(self.crypto_stop_loss)
        crypto_stop_loss_layout.addStretch()
        crypto_params_layout.addLayout(crypto_stop_loss_layout)

        # Take profit (crypto)
        crypto_take_profit_layout = QHBoxLayout()
        crypto_take_profit_layout.addWidget(QLabel("Take Profit (%):"))
        self.crypto_take_profit = QDoubleSpinBox()
        self.crypto_take_profit.setRange(1.0, 100.0)
        self.crypto_take_profit.setSingleStep(1.0)
        self.crypto_take_profit.setDecimals(0)
        self.crypto_take_profit.setValue(25)
        crypto_take_profit_layout.addWidget(self.crypto_take_profit)
        crypto_take_profit_layout.addStretch()
        crypto_params_layout.addLayout(crypto_take_profit_layout)

        # Lookback days
        crypto_lookback_layout = QHBoxLayout()
        crypto_lookback_layout.addWidget(QLabel("Lookback Days:"))
        self.crypto_lookback_days = QSpinBox()
        self.crypto_lookback_days.setRange(30, 1095)
        self.crypto_lookback_days.setSingleStep(30)
        self.crypto_lookback_days.setValue(365)
        crypto_lookback_layout.addWidget(self.crypto_lookback_days)
        crypto_lookback_layout.addStretch()
        crypto_params_layout.addLayout(crypto_lookback_layout)

        # Time in force
        crypto_tif_layout = QHBoxLayout()
        crypto_tif_layout.addWidget(QLabel("Time in Force:"))
        self.crypto_time_in_force = QComboBox()
        self.crypto_time_in_force.addItems(["gtc", "day", "ioc", "fok"])
        self.crypto_time_in_force.setCurrentText("gtc")
        crypto_tif_layout.addWidget(self.crypto_time_in_force)
        crypto_tif_layout.addStretch()
        crypto_params_layout.addLayout(crypto_tif_layout)

        crypto_layout.addLayout(crypto_params_layout)

        crypto_group.setLayout(crypto_layout)
        layout.addWidget(crypto_group)

        # Buttons
        buttons_layout = QHBoxLayout()

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_config)
        buttons_layout.addWidget(save_btn)

        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self.load_config)
        buttons_layout.addWidget(reload_btn)

        buttons_layout.addStretch()

        layout.addLayout(buttons_layout)

        # Info label
        info_label = QLabel("⚠️ Note: Bot must be restarted for changes to take effect")
        info_label.setStyleSheet("color: orange;")
        layout.addWidget(info_label)

        layout.addStretch()

    def toggle_crypto_settings(self, state):
        """Enable/disable crypto settings based on checkbox"""
        enabled = state == Qt.CheckState.Checked.value
        self.crypto_tickers_text.setEnabled(enabled)
        self.crypto_max_position.setEnabled(enabled)
        self.crypto_risk_multiplier.setEnabled(enabled)
        self.crypto_stop_loss.setEnabled(enabled)
        self.crypto_take_profit.setEnabled(enabled)
        self.crypto_lookback_days.setEnabled(enabled)
        self.crypto_time_in_force.setEnabled(enabled)

    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            # Populate UI
            tickers = self.config.get('tickers', [])
            self.tickers_text.setPlainText('\n'.join(tickers))

            self.risk_per_trade.setValue(self.config.get('risk_per_trade_pct', 0.2))
            self.max_position.setValue(self.config.get('max_position_pct', 5.0))
            self.stop_loss.setValue(self.config.get('stop_loss_pct', 0.05))
            self.take_profit.setValue(self.config.get('take_profit_pct', 0.1))
            self.buying_power.setValue(self.config.get('buying_power_pct', 50))

            # Load crypto settings
            crypto_config = self.config.get('crypto', {})
            crypto_enabled = crypto_config.get('enabled', False)
            self.crypto_enabled.setChecked(crypto_enabled)

            crypto_tickers = crypto_config.get('tickers', [])
            self.crypto_tickers_text.setPlainText('\n'.join(crypto_tickers))

            self.crypto_max_position.setValue(crypto_config.get('max_position_pct', 2.5))
            self.crypto_risk_multiplier.setValue(crypto_config.get('risk_multiplier', 1.5))
            self.crypto_stop_loss.setValue(crypto_config.get('stop_loss_pct', 15))
            self.crypto_take_profit.setValue(crypto_config.get('take_profit_pct', 25))
            self.crypto_lookback_days.setValue(crypto_config.get('lookback_days', 365))
            self.crypto_time_in_force.setCurrentText(crypto_config.get('time_in_force', 'gtc'))

            # Trigger toggle to enable/disable crypto fields
            self.toggle_crypto_settings(Qt.CheckState.Checked.value if crypto_enabled else Qt.CheckState.Unchecked.value)

            logger.info("Configuration loaded")

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")

    def save_config(self):
        """Save configuration to file"""
        try:
            # Update config from UI
            tickers_text = self.tickers_text.toPlainText()
            tickers = [t.strip() for t in tickers_text.split('\n') if t.strip()]

            self.config['tickers'] = tickers
            self.config['risk_per_trade_pct'] = self.risk_per_trade.value()
            self.config['max_position_pct'] = self.max_position.value()
            self.config['stop_loss_pct'] = self.stop_loss.value()
            self.config['take_profit_pct'] = self.take_profit.value()
            self.config['buying_power_pct'] = int(self.buying_power.value())

            # Save crypto settings
            crypto_tickers_text = self.crypto_tickers_text.toPlainText()
            crypto_tickers = [t.strip() for t in crypto_tickers_text.split('\n') if t.strip()]

            if 'crypto' not in self.config:
                self.config['crypto'] = {}

            self.config['crypto']['enabled'] = self.crypto_enabled.isChecked()
            self.config['crypto']['tickers'] = crypto_tickers
            self.config['crypto']['max_position_pct'] = self.crypto_max_position.value()
            self.config['crypto']['risk_multiplier'] = self.crypto_risk_multiplier.value()
            self.config['crypto']['stop_loss_pct'] = self.crypto_stop_loss.value() / 100  # Convert to decimal
            self.config['crypto']['take_profit_pct'] = self.crypto_take_profit.value() / 100  # Convert to decimal
            self.config['crypto']['lookback_days'] = self.crypto_lookback_days.value()
            self.config['crypto']['time_in_force'] = self.crypto_time_in_force.currentText()
            self.config['crypto']['min_qty'] = 0.0001
            self.config['crypto']['risk_per_trade_pct'] = 0.15
            self.config['crypto']['trading_hours'] = {'enabled': False}

            # Save to file
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)

            logger.info("Configuration saved")
            QMessageBox.information(
                self,
                "Success",
                "Settings saved successfully!\n\nRestart the bot for changes to take effect."
            )

        except Exception as e:
            logger.error(f"Error saving config: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save config: {e}")

    def closeEvent(self, event):
        """Handle window close"""
        # Just hide the window instead of closing
        event.ignore()
        self.hide()
