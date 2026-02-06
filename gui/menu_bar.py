"""
Menu Bar Application - Main GUI entry point
Native macOS menu bar icon with popup menu
"""
import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QSystemTrayIcon, QMenu
)
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QAction
from PyQt6.QtCore import QTimer, Qt
from ipc_protocol import IPCClient
import logging

logger = logging.getLogger(__name__)


class TradingBotMenuBar:
    """Menu bar application for trading bot"""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)  # Keep running when windows close

        self.ipc_client = IPCClient()
        self.tray_icon = None
        self.menu = None
        self.status_timer = QTimer()

        # Windows (imported lazily to avoid circular imports)
        self.dashboard_window = None
        self.logs_window = None
        self.settings_window = None

        # Current state
        self.bot_status = "stopped"
        self.is_trading = False

        self.setup_ui()

    def setup_ui(self):
        """Setup the menu bar UI"""
        # Create tray icon
        self.tray_icon = QSystemTrayIcon()
        self.tray_icon.setIcon(self._create_status_icon("gray"))
        self.tray_icon.setToolTip("Trading Bot - Stopped")

        # Create menu
        self.create_menu()

        # Show tray icon
        self.tray_icon.show()

        # Start status update timer
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)  # Update every 5 seconds

        # Initial status check
        self.update_status()

    def create_menu(self):
        """Create the popup menu"""
        self.menu = QMenu()

        # Status label (non-clickable)
        self.status_action = QAction("🔴 Bot Status: Stopped")
        self.status_action.setEnabled(False)
        self.menu.addAction(self.status_action)

        # Heartbeat label (non-clickable)
        self.heartbeat_action = QAction("⏱ Last heartbeat: —")
        self.heartbeat_action.setEnabled(False)
        self.menu.addAction(self.heartbeat_action)

        self.menu.addSeparator()

        # Start/Stop actions
        self.start_action = QAction("▶️  Start Bot")
        self.start_action.triggered.connect(self.start_bot)
        self.menu.addAction(self.start_action)

        self.stop_action = QAction("⏸️  Stop Bot")
        self.stop_action.triggered.connect(self.stop_bot)
        self.stop_action.setEnabled(False)
        self.menu.addAction(self.stop_action)

        # Run-now action
        self.run_now_action = QAction("⚡ Run Now")
        self.run_now_action.triggered.connect(self.run_now)
        self.run_now_action.setEnabled(False)
        self.menu.addAction(self.run_now_action)

        self.menu.addSeparator()

        # Keep references to actions as attributes to avoid PyQt GC edge-cases
        self.dashboard_action = QAction("📊 Dashboard")
        self.dashboard_action.triggered.connect(self.show_dashboard)
        self.menu.addAction(self.dashboard_action)

        # Positions action
        self.positions_action = QAction("📈 Positions")
        self.positions_action.triggered.connect(self.show_positions)
        self.menu.addAction(self.positions_action)

        # Signals action
        self.signals_action = QAction("📋 Today's Signals")
        self.signals_action.triggered.connect(self.show_signals)
        self.menu.addAction(self.signals_action)

        # Logs action
        self.logs_action = QAction("📄 Live Logs")
        self.logs_action.triggered.connect(self.show_logs)
        self.menu.addAction(self.logs_action)

        # Settings action
        self.settings_action = QAction("⚙️  Settings")
        self.settings_action.triggered.connect(self.show_settings)
        self.menu.addAction(self.settings_action)

        # Backtest action
        self.backtest_action = QAction("🧪 Train/Backtest")
        self.backtest_action.triggered.connect(self.show_backtest)
        self.menu.addAction(self.backtest_action)

        # Chat action
        self.chat_action = QAction("💬 Ask Dexter")
        self.chat_action.triggered.connect(self.show_chat)
        self.menu.addAction(self.chat_action)

        self.menu.addSeparator()

        # Quit action
        self.quit_action = QAction("❌ Quit")
        self.quit_action.triggered.connect(self.quit_application)
        self.menu.addAction(self.quit_action)

        self.tray_icon.setContextMenu(self.menu)

    def _create_status_icon(self, color: str) -> QIcon:
        """Create a colored circle icon"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Color mapping
        color_map = {
            "green": QColor(76, 175, 80),    # Running
            "yellow": QColor(255, 193, 7),   # Idle
            "red": QColor(244, 67, 54),      # Stopped/Error
            "gray": QColor(158, 158, 158),   # Unknown
            "blue": QColor(33, 150, 243)     # Trading
        }

        painter.setBrush(color_map.get(color, color_map["gray"]))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, 24, 24)
        painter.end()

        return QIcon(pixmap)

    def update_status(self):
        """Update bot status from service"""
        if not self.ipc_client.is_running():
            self.bot_status = "stopped"
            self.status_action.setText("🔴 Bot Service: Not Running")
            self.heartbeat_action.setText("⏱ Last heartbeat: —")
            self.tray_icon.setIcon(self._create_status_icon("red"))
            self.tray_icon.setToolTip("Trading Bot - Service Not Running")
            self.start_action.setEnabled(False)
            self.stop_action.setEnabled(False)
            self.run_now_action.setEnabled(False)
            return

        # Get status from service
        response = self.ipc_client.send_command({"command": "get_status"})

        # IPC errors are returned as {"error": "..."}; normal status includes error=None
        if response.get("error"):
            self.bot_status = "error"
            self.status_action.setText(f"⚠️  Bot Status: Error")
            self.tray_icon.setIcon(self._create_status_icon("red"))
            self.tray_icon.setToolTip(f"Trading Bot - Error: {response['error']}")
            return

        status = response.get("status", "unknown")
        self.bot_status = status
        running = response.get("running", False)
        next_exec = response.get("next_execution")
        last_heartbeat = response.get("last_heartbeat")
        if last_heartbeat:
            self.heartbeat_action.setText(f"⏱ Last heartbeat: {last_heartbeat}")
        else:
            self.heartbeat_action.setText("⏱ Last heartbeat: —")

        # Update status text and icon
        if status == "trading":
            icon_color = "blue"
            status_text = "🔵 Bot Status: Trading"
            tooltip = "Trading Bot - Executing Trades"
        elif status == "running":
            icon_color = "green"
            status_text = f"🟢 Bot Status: Running (Next: {next_exec or 'N/A'})"
            tooltip = f"Trading Bot - Running (Next: {next_exec or 'N/A'})"
        elif status == "idle":
            icon_color = "yellow"
            status_text = "🟡 Bot Status: Idle"
            tooltip = "Trading Bot - Idle"
        elif status == "error":
            icon_color = "red"
            error_msg = response.get("error", "Unknown error")
            status_text = "🔴 Bot Status: Error"
            tooltip = f"Trading Bot - Error: {error_msg}"
        else:
            icon_color = "gray"
            status_text = "⚪ Bot Status: Stopped"
            tooltip = "Trading Bot - Stopped"

        self.status_action.setText(status_text)
        self.tray_icon.setIcon(self._create_status_icon(icon_color))
        self.tray_icon.setToolTip(tooltip)

        # Update button states
        if running:
            self.start_action.setEnabled(False)
            self.stop_action.setEnabled(True)
            self.run_now_action.setEnabled(True)
        else:
            self.start_action.setEnabled(True)
            self.stop_action.setEnabled(False)
            self.run_now_action.setEnabled(False)

    def run_now(self):
        """Trigger an immediate run (useful for testing without waiting for schedule)."""
        response = self.ipc_client.send_command({"command": "run_now"})

        if response.get("success"):
            logger.info("Run-now triggered successfully")
            self.tray_icon.showMessage(
                "Trading Bot",
                "Run-now triggered",
                QSystemTrayIcon.MessageIcon.Information,
                3000
            )
        else:
            msg = response.get("message") or response.get("error") or "Unknown error"
            logger.error(f"Run-now failed: {msg}")
            self.tray_icon.showMessage(
                "Trading Bot",
                f"Run-now failed: {msg}",
                QSystemTrayIcon.MessageIcon.Critical,
                5000
            )
        self.update_status()

    def start_bot(self):
        """Start the trading bot"""
        response = self.ipc_client.send_command({"command": "start"})

        if response.get("success"):
            logger.info("Bot started successfully")
            self.tray_icon.showMessage(
                "Trading Bot",
                "Bot started successfully",
                QSystemTrayIcon.MessageIcon.Information,
                3000
            )
        else:
            error_msg = response.get("message", "Unknown error")
            logger.error(f"Failed to start bot: {error_msg}")
            self.tray_icon.showMessage(
                "Trading Bot",
                f"Failed to start: {error_msg}",
                QSystemTrayIcon.MessageIcon.Critical,
                5000
            )

        self.update_status()

    def stop_bot(self):
        """Stop the trading bot"""
        response = self.ipc_client.send_command({"command": "stop"})

        if response.get("success"):
            logger.info("Bot stopped successfully")
            self.tray_icon.showMessage(
                "Trading Bot",
                "Bot stopped successfully",
                QSystemTrayIcon.MessageIcon.Information,
                3000
            )
        else:
            error_msg = response.get("message", "Unknown error")
            logger.error(f"Failed to stop bot: {error_msg}")

        self.update_status()

    def show_dashboard(self):
        """Show the dashboard window"""
        from gui.dashboard_window import DashboardWindow

        if self.dashboard_window is None:
            self.dashboard_window = DashboardWindow(self.ipc_client)

        self.dashboard_window.show()
        self.dashboard_window.raise_()
        self.dashboard_window.activateWindow()

    def show_positions(self):
        """Show positions in dashboard"""
        self.show_dashboard()
        if self.dashboard_window is not None:
            self.dashboard_window.focus_positions()

    def show_signals(self):
        """Show signals in dashboard"""
        self.show_dashboard()
        if self.dashboard_window is not None:
            self.dashboard_window.focus_signals()

    def show_logs(self):
        """Show the logs window"""
        from gui.logs_window import LogsWindow

        if self.logs_window is None:
            self.logs_window = LogsWindow()

        self.logs_window.show()
        self.logs_window.raise_()
        self.logs_window.activateWindow()

    def show_settings(self):
        """Show the settings window"""
        from gui.settings_window import SettingsWindow

        if self.settings_window is None:
            self.settings_window = SettingsWindow()

        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def show_backtest(self):
        """Show the train/backtest window"""
        from gui.backtest_window import BacktestWindow

        if getattr(self, "backtest_window", None) is None:
            self.backtest_window = BacktestWindow(self.ipc_client)

        self.backtest_window.show()
        self.backtest_window.raise_()
        self.backtest_window.activateWindow()

    def show_chat(self):
        """Show the Dexter chat window"""
        from gui.chat_window import ChatWindow

        if getattr(self, "chat_window", None) is None:
            self.chat_window = ChatWindow(self.ipc_client)

        self.chat_window.show()
        self.chat_window.raise_()
        self.chat_window.activateWindow()

    def quit_application(self):
        """Quit the application"""
        # Ask if user wants to stop bot too
        from PyQt6.QtWidgets import QMessageBox

        if self.bot_status in ["running", "trading"]:
            reply = QMessageBox.question(
                None,
                "Quit Application",
                "The bot is currently running. Stop the bot and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.stop_bot()

        self.app.quit()

    def run(self):
        """Start the application event loop"""
        return self.app.exec()


def main():
    """Main entry point for menu bar app"""
    app = TradingBotMenuBar()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
