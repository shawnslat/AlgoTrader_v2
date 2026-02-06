#!/usr/bin/env python3
"""
Unified Trading Bot Launcher

Starts the bot service in the background and launches the GUI.
When the user quits, both the GUI and the bot service are stopped.

This is a single entry point replacing:
- 1_start_bot_service.sh
- 2_start_gui.sh
"""
import sys
import os
import subprocess
import signal
import time
import atexit
from pathlib import Path

# Set Qt environment variables before importing PyQt6
os.environ['QT_MAC_WANTS_LAYER'] = '1'

# Change to the project directory (where this script lives)
PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

# Determine Python interpreter
if (PROJECT_DIR / ".venv/bin/python").exists():
    PYTHON = str(PROJECT_DIR / ".venv/bin/python")
else:
    PYTHON = "python3"
    print("Warning: .venv not found; using system python3")

# Global reference to bot service process
bot_process = None


def cleanup():
    """Cleanup function to stop bot service when app exits"""
    global bot_process
    if bot_process and bot_process.poll() is None:
        print("Stopping bot service...")
        bot_process.terminate()
        try:
            bot_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bot_process.kill()
        print("Bot service stopped.")


def start_bot_service():
    """Start the bot service as a background process"""
    global bot_process

    # Clean up old socket
    socket_path = Path("/tmp/trader_bot.sock")
    if socket_path.exists():
        socket_path.unlink()

    # Ensure logs directory exists
    (PROJECT_DIR / "logs").mkdir(exist_ok=True)

    print("Starting bot service in background...")

    # Start bot_service.py as subprocess
    bot_process = subprocess.Popen(
        [PYTHON, str(PROJECT_DIR / "bot_service.py")],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # Don't capture stdin so it doesn't block
        stdin=subprocess.DEVNULL,
    )

    # Register cleanup handler
    atexit.register(cleanup)

    # Wait for socket to be ready (up to 30 seconds)
    print("Waiting for bot service to initialize...")
    for i in range(60):  # 30 seconds (0.5s * 60)
        if socket_path.exists():
            print("Bot service ready!")
            return True
        if bot_process.poll() is not None:
            # Process died
            output = bot_process.stdout.read().decode('utf-8', errors='replace')
            print(f"Bot service failed to start:\n{output}")
            return False
        time.sleep(0.5)

    print("Bot service took too long to start")
    return False


def main():
    """Main entry point"""
    print("=" * 50)
    print("  Trading Bot - Unified Launcher")
    print("=" * 50)
    print()

    # Check if bot service is already running
    from ipc_protocol import IPCClient
    client = IPCClient()

    if client.is_running():
        print("Bot service already running (using existing instance)")
        bot_already_running = True
    else:
        # Start bot service
        if not start_bot_service():
            print("Failed to start bot service. Exiting.")
            sys.exit(1)
        bot_already_running = False

    # Now import PyQt6 and create the GUI
    from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QMessageBox
    from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QAction
    from PyQt6.QtCore import QTimer, Qt

    app = QApplication(sys.argv)
    app.setApplicationName("Trading Bot")
    app.setQuitOnLastWindowClosed(False)

    # Create tray icon with color
    def create_icon(color_name):
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        colors = {
            "green": QColor(76, 175, 80),
            "yellow": QColor(255, 193, 7),
            "red": QColor(244, 67, 54),
            "gray": QColor(158, 158, 158),
            "blue": QColor(33, 150, 243),
        }

        painter.setBrush(colors.get(color_name, colors["gray"]))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, 24, 24)
        painter.end()

        return QIcon(pixmap)

    # Check if system tray is available
    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("ERROR: System tray not available on this system")
        sys.exit(1)

    # Create tray icon
    tray = QSystemTrayIcon()
    tray.setIcon(create_icon("yellow"))
    tray.setToolTip("Trading Bot - Loading...")

    # Create menu
    menu = QMenu()

    status_action = QAction("🟡 Bot Status: Loading...")
    status_action.setEnabled(False)
    menu.addAction(status_action)

    heartbeat_action = QAction("⏱ Last heartbeat: —")
    heartbeat_action.setEnabled(False)
    menu.addAction(heartbeat_action)

    menu.addSeparator()

    start_action = QAction("▶️  Start Bot")
    stop_action = QAction("⏸️  Stop Bot")
    stop_action.setEnabled(False)

    menu.addAction(start_action)
    menu.addAction(stop_action)

    run_now_action = QAction("⚡ Run Now")
    run_now_action.setEnabled(False)
    menu.addAction(run_now_action)

    restart_action = QAction("🔄 Restart Bot")
    restart_action.setEnabled(False)
    menu.addAction(restart_action)

    menu.addSeparator()

    dashboard_action = QAction("📊 Dashboard")
    logs_action = QAction("📄 Live Logs")
    settings_action = QAction("⚙️  Settings")
    backtest_action = QAction("🧪 Train/Backtest")
    chat_action = QAction("💬 Ask Dexter")

    menu.addAction(dashboard_action)
    menu.addAction(logs_action)
    menu.addAction(settings_action)
    menu.addAction(backtest_action)
    menu.addAction(chat_action)

    menu.addSeparator()

    quit_action = QAction("❌ Quit")
    menu.addAction(quit_action)

    tray.setContextMenu(menu)
    tray.show()

    # Window references
    windows = {}

    # Status update function
    def update_status():
        try:
            response = client.send_command({"command": "get_status"})

            if response.get("error"):
                status_action.setText("🔴 Bot Status: Error")
                heartbeat_action.setText("⏱ Last heartbeat: —")
                tray.setIcon(create_icon("red"))
                tray.setToolTip(f"Trading Bot - Error: {response['error']}")
                start_action.setEnabled(False)
                stop_action.setEnabled(False)
                run_now_action.setEnabled(False)
                restart_action.setEnabled(False)
            else:
                status = response.get("status", "unknown")
                running = response.get("running", False)
                next_exec = response.get("next_execution", "N/A")
                hb = response.get("last_heartbeat")
                heartbeat_action.setText(f"⏱ Last heartbeat: {hb or '—'}")

                if status == "running":
                    status_action.setText(f"🟢 Bot Status: Running (Next: {next_exec})")
                    tray.setIcon(create_icon("green"))
                    tray.setToolTip(f"Trading Bot - Running (Next: {next_exec})")
                    start_action.setEnabled(False)
                    stop_action.setEnabled(True)
                    run_now_action.setEnabled(True)
                    restart_action.setEnabled(True)
                elif status == "trading":
                    status_action.setText("🔵 Bot Status: Trading")
                    tray.setIcon(create_icon("blue"))
                    tray.setToolTip("Trading Bot - Trading")
                    start_action.setEnabled(False)
                    stop_action.setEnabled(True)
                    run_now_action.setEnabled(True)
                    restart_action.setEnabled(True)
                elif status == "error":
                    error_msg = response.get("error", "Unknown error")
                    status_action.setText("🔴 Bot Status: Error")
                    tray.setIcon(create_icon("red"))
                    tray.setToolTip(f"Trading Bot - Error: {error_msg}")
                    start_action.setEnabled(False)
                    stop_action.setEnabled(False)
                    run_now_action.setEnabled(False)
                    restart_action.setEnabled(True)  # Allow restart on error
                elif status == "stopped":
                    status_action.setText("⚪ Bot Status: Stopped")
                    tray.setIcon(create_icon("gray"))
                    tray.setToolTip("Trading Bot - Stopped")
                    start_action.setEnabled(True)
                    stop_action.setEnabled(False)
                    run_now_action.setEnabled(False)
                    restart_action.setEnabled(True)
                else:
                    status_action.setText("🟡 Bot Status: Idle")
                    tray.setIcon(create_icon("yellow"))
                    tray.setToolTip("Trading Bot - Idle")
                    start_action.setEnabled(True)
                    stop_action.setEnabled(False)
                    run_now_action.setEnabled(False)
                    restart_action.setEnabled(True)
        except Exception as e:
            status_action.setText("🔴 Bot Status: Disconnected")
            heartbeat_action.setText("⏱ Last heartbeat: —")
            tray.setIcon(create_icon("red"))
            tray.setToolTip("Trading Bot - Disconnected")
            restart_action.setEnabled(True)  # Allow restart when disconnected
            print(f"Error updating status: {e}")

    # Start/stop handlers
    def start_bot():
        response = client.send_command({"command": "start"})
        if response.get("success"):
            tray.showMessage("Trading Bot", "Bot started successfully", QSystemTrayIcon.MessageIcon.Information, 3000)
        update_status()

    def stop_bot():
        response = client.send_command({"command": "stop"})
        if response.get("success"):
            tray.showMessage("Trading Bot", "Bot stopped successfully", QSystemTrayIcon.MessageIcon.Information, 3000)
        update_status()

    def run_now():
        response = client.send_command({"command": "run_now"})
        if response.get("success"):
            tray.showMessage("Trading Bot", "Run-now triggered", QSystemTrayIcon.MessageIcon.Information, 3000)
        else:
            msg = response.get("message") or response.get("error") or "Unknown error"
            tray.showMessage("Trading Bot", f"Run-now failed: {msg}", QSystemTrayIcon.MessageIcon.Critical, 5000)
        update_status()

    def restart_bot():
        """Restart the bot service to apply config changes"""
        global bot_process

        tray.showMessage("Trading Bot", "Restarting bot service...", QSystemTrayIcon.MessageIcon.Information, 2000)
        status_action.setText("🟡 Bot Status: Restarting...")
        tray.setIcon(create_icon("yellow"))

        # Stop the trading loop
        client.send_command({"command": "stop"})

        # Kill the bot service process if we started it
        if bot_process and bot_process.poll() is None:
            bot_process.terminate()
            try:
                bot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bot_process.kill()
                bot_process.wait()

        # Wait a moment for cleanup
        time.sleep(1)

        # Start bot service again
        if start_bot_service():
            # Wait for service to be ready
            time.sleep(2)
            # Start the trading loop
            client.send_command({"command": "start"})
            tray.showMessage("Trading Bot", "Bot restarted successfully!", QSystemTrayIcon.MessageIcon.Information, 3000)
        else:
            tray.showMessage("Trading Bot", "Failed to restart bot service", QSystemTrayIcon.MessageIcon.Critical, 5000)

        update_status()

    # Main window (tabbed interface)
    main_window = None

    def show_main_window(tab_index=0):
        """Show the main tabbed window"""
        nonlocal main_window
        try:
            from gui.main_window import MainWindow
            if main_window is None:
                main_window = MainWindow(client)
            main_window.tabs.setCurrentIndex(tab_index)
            main_window.show()
            main_window.raise_()
            main_window.activateWindow()
        except Exception as e:
            print(f"Error opening main window: {e}")
            import traceback
            traceback.print_exc()

    def show_dashboard():
        show_main_window(0)  # Dashboard tab

    def show_logs():
        show_main_window(1)  # Logs tab

    def show_settings():
        show_main_window(2)  # Settings tab

    def show_backtest():
        show_main_window(3)  # Backtest tab

    def show_chat():
        show_main_window(4)  # Chat tab

    def quit_application():
        """Quit application and stop bot service"""
        global bot_process

        # Check if bot is running (trading loop active)
        response = client.send_command({"command": "get_status"})
        bot_running = response.get("running", False)

        if bot_running:
            reply = QMessageBox.question(
                None,
                "Quit Trading Bot",
                "The trading bot is currently running.\n\nStop the bot and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # Stop the trading loop
            client.send_command({"command": "stop"})

        # If we started the bot service, kill it
        if bot_process and bot_process.poll() is None:
            print("Stopping bot service...")
            bot_process.terminate()
            try:
                bot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bot_process.kill()
            print("Bot service stopped.")

        app.quit()

    # Connect actions
    start_action.triggered.connect(start_bot)
    stop_action.triggered.connect(stop_bot)
    run_now_action.triggered.connect(run_now)
    restart_action.triggered.connect(restart_bot)
    dashboard_action.triggered.connect(show_dashboard)
    logs_action.triggered.connect(show_logs)
    settings_action.triggered.connect(show_settings)
    backtest_action.triggered.connect(show_backtest)
    chat_action.triggered.connect(show_chat)
    quit_action.triggered.connect(quit_application)

    # Setup status timer
    status_timer = QTimer()
    status_timer.timeout.connect(update_status)
    status_timer.start(5000)  # Update every 5 seconds

    # Initial update
    update_status()

    print()
    print("Trading Bot is now running!")
    print("Look for the menu bar icon in the top-right of your screen.")
    print()

    # Run app
    exit_code = app.exec()

    # Cleanup will be called by atexit
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
