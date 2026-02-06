"""
Chat Window - Ask questions to Dexter (via bot_service IPC).
"""
import threading
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QPushButton, QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt

from ipc_protocol import IPCClient


class _ChatSignals(QObject):
    finished = pyqtSignal(dict)


class ChatWindow(QMainWindow):
    def __init__(self, ipc_client: IPCClient):
        super().__init__()
        self.ipc_client = ipc_client
        self.signals = _ChatSignals()
        self.signals.finished.connect(self._on_response)
        self.setWindowTitle("Trading Bot - Ask Dexter")
        self.setGeometry(140, 140, 900, 650)

        self._build_ui()

    def _build_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)

        top = QHBoxLayout()
        self.status_label = QLabel("Ready")
        top.addWidget(self.status_label)
        top.addStretch()
        self.include_context = QCheckBox("Include account/positions/signals")
        self.include_context.setChecked(True)
        top.addWidget(self.include_context)
        layout.addLayout(top)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(self.chat_history)

        bottom = QHBoxLayout()
        self.input_box = QTextEdit()
        self.input_box.setFixedHeight(90)
        bottom.addWidget(self.input_box, stretch=1)

        right = QVBoxLayout()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send)
        right.addWidget(self.send_btn)

        self.use_tickers_btn = QPushButton("Use Dexter Tickrs Today")
        self.use_tickers_btn.clicked.connect(self.use_dexter_tickers)
        right.addWidget(self.use_tickers_btn)

        self.bias_btn = QPushButton("Generate Dexter Bias")
        self.bias_btn.clicked.connect(self.generate_bias)
        right.addWidget(self.bias_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.chat_history.clear)
        right.addWidget(self.clear_btn)

        right.addStretch()
        bottom.addLayout(right)

        layout.addLayout(bottom)

    def _append(self, role: str, text: str):
        self.chat_history.append(f"<b>{role}:</b> {text}")
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def _set_busy(self, busy: bool, status: str):
        self.status_label.setText(status)
        self.send_btn.setEnabled(not busy)

    def send(self):
        query = (self.input_box.toPlainText() or "").strip()
        if not query:
            return
        self.input_box.clear()
        self._append("You", query)
        self._set_busy(True, "Asking Dexter…")

        include_ctx = self.include_context.isChecked()

        def worker():
            resp = self.ipc_client.send_command(
                {"command": "dexter_chat", "query": query, "include_context": include_ctx},
                timeout=180.0,
            )
            self.signals.finished.emit(resp)

        threading.Thread(target=worker, daemon=True).start()

    def _on_response(self, resp: dict):
        if resp.get("error"):
            self._set_busy(False, "Error")
            self._append("Error", str(resp.get("error")))
            return
        self._set_busy(False, "Ready")
        self._append("Dexter", str(resp.get("answer", "")))

    def use_dexter_tickers(self):
        resp = self.ipc_client.send_command({"command": "dexter_update_tickers"}, timeout=5.0)
        if resp.get("error"):
            QMessageBox.critical(self, "Dexter Tickrs", resp.get("error"))
            return
        if resp.get("success") is False:
            QMessageBox.information(self, "Dexter Tickrs", resp.get("message", "Already running."))
            return
        QMessageBox.information(self, "Dexter Tickrs", "Ticker update started. It will update tickers_auto.json when finished.")

    def generate_bias(self):
        resp = self.ipc_client.send_command({"command": "dexter_generate_bias"}, timeout=5.0)
        if resp.get("error"):
            QMessageBox.critical(self, "Dexter Bias", resp.get("error"))
            return
        if resp.get("success") is False:
            QMessageBox.information(self, "Dexter Bias", resp.get("message", "Already running."))
            return
        QMessageBox.information(self, "Dexter Bias", "Bias generation started. It will write dexter_bias.json when finished.")

    def closeEvent(self, event):
        event.ignore()
        self.hide()
