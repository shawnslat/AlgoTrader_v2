"""
Logs Window - Real-time log viewer
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QComboBox, QLabel
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QTextCursor
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LogsWindow(QMainWindow):
    """Window for viewing real-time logs"""

    def __init__(self):
        super().__init__()
        # Use absolute paths based on this file's location
        base_dir = Path(__file__).resolve().parent.parent  # Project root (absolute)
        self.primary_log_file = base_dir / "logs" / "bot_service.log"  # Primary is bot_service.log
        self.fallback_log_file = base_dir / "logs" / "master_trading_bot.log"

        # Choose file that exists AND has content
        primary_ok = self.primary_log_file.exists() and self.primary_log_file.stat().st_size > 0
        fallback_ok = self.fallback_log_file.exists() and self.fallback_log_file.stat().st_size > 0

        if primary_ok:
            self.log_file = self.primary_log_file
        elif fallback_ok:
            self.log_file = self.fallback_log_file
        else:
            self.log_file = self.primary_log_file  # Default, will show "no logs"

        # Start at end of file (only show new logs, not 57MB of history)
        self.last_position = 0
        if self.log_file.exists() and self.log_file.stat().st_size > 0:
            self.last_position = max(0, self.log_file.stat().st_size - 50000)  # Last ~50KB

        self.setup_ui()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_logs)
        self.refresh_timer.start(2000)  # Refresh every 2 seconds

        # Initial load
        self.refresh_logs()

    def setup_ui(self):
        """Setup the logs window UI"""
        self.setWindowTitle("Trading Bot - Live Logs")
        self.setGeometry(100, 100, 900, 600)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Controls
        controls_layout = QHBoxLayout()

        # File label
        self.file_label = QLabel(f"File: {self.log_file.name}")
        controls_layout.addWidget(self.file_label)

        # Filter combo
        controls_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "INFO", "WARNING", "ERROR", "DEBUG"])
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        controls_layout.addWidget(self.filter_combo)

        controls_layout.addStretch()

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_logs)
        controls_layout.addWidget(clear_btn)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_logs)
        controls_layout.addWidget(refresh_btn)

        layout.addLayout(controls_layout)

        # Log text area - dark theme
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

    def refresh_logs(self):
        """Refresh logs from file"""
        # Switch to fallback if primary isn't present or is empty
        primary_exists = self.primary_log_file.exists() and self.primary_log_file.stat().st_size > 0
        fallback_exists = self.fallback_log_file.exists() and self.fallback_log_file.stat().st_size > 0

        if primary_exists:
            if self.log_file != self.primary_log_file:
                self.log_file = self.primary_log_file
                self.file_label.setText(f"File: {self.log_file.name}")
                self.log_text.clear()
                self.last_position = 0
        elif fallback_exists:
            if self.log_file != self.fallback_log_file:
                self.log_file = self.fallback_log_file
                self.file_label.setText(f"File: {self.log_file.name}")
                self.log_text.clear()
                self.last_position = 0
        else:
            self.file_label.setText("File: (no logs yet)")
            return

        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='replace') as f:
                # Seek to last position
                f.seek(self.last_position)

                # If starting mid-file, skip partial first line
                if self.last_position > 0 and self.log_text.toPlainText() == "":
                    f.readline()  # Skip potentially partial line

                # Read new lines
                new_lines = f.readlines()

                if new_lines:
                    # Update position
                    self.last_position = f.tell()

                    # Append new lines
                    for line in new_lines:
                        self.append_log_line(line.rstrip())

                    # Auto-scroll to bottom
                    cursor = self.log_text.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    self.log_text.setTextCursor(cursor)

                # Update file label with size info
                size_mb = self.log_file.stat().st_size / (1024 * 1024)
                self.file_label.setText(f"File: {self.log_file.name} ({size_mb:.1f} MB)")

        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            self.file_label.setText(f"Error: {e}")

    def append_log_line(self, line: str):
        """Append a log line with color coding"""
        # Apply current filter
        current_filter = self.filter_combo.currentText()
        if current_filter != "All":
            if f"[{current_filter}]" not in line:
                return

        # Add line to text widget
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Color based on level (light colors for dark background)
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

    def apply_filter(self):
        """Apply log level filter"""
        # Re-read entire file with filter
        self.log_text.clear()
        self.last_position = 0
        self.refresh_logs()

    def clear_logs(self):
        """Clear the log display"""
        self.log_text.clear()

    def closeEvent(self, event):
        """Handle window close"""
        # Just hide the window instead of closing
        event.ignore()
        self.hide()
