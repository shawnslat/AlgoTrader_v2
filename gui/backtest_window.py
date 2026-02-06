"""
Backtest Window - Run download/train/backtest and view artifacts.
"""
from pathlib import Path
import os
import pandas as pd
import subprocess
import sys

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem,
    QMessageBox, QScrollArea
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

from ipc_protocol import IPCClient
import logging

logger = logging.getLogger(__name__)


class BacktestWindow(QMainWindow):
    def __init__(self, ipc_client: IPCClient):
        super().__init__()
        self.ipc_client = ipc_client

        self.setWindowTitle("Trading Bot - Train & Backtest")
        self.setGeometry(120, 120, 1100, 800)

        os.makedirs("artifacts", exist_ok=True)

        self.report_path = Path("artifacts/classification_report.txt")
        self.cm_path = Path("artifacts/confusion_matrix.png")
        self.bt_png_path = Path("artifacts/backtesting_results.png")
        self.bt_csv_path = Path("artifacts/backtesting_results.csv")
        self.validation_report_path = Path("artifacts/validation_report.txt")
        self.oos_comparison_path = Path("artifacts/out_of_sample_comparison.png")
        self.feature_importance_path = Path("artifacts/feature_importance.png")

        self._build_ui()

        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self.refresh_status)
        self.poll_timer.start(2000)

        self.refresh_status()
        self.load_artifacts()

    def _build_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)

        header = QHBoxLayout()
        self.status_label = QLabel("Status: —")
        header.addWidget(self.status_label)

        header.addStretch()

        self.run_btn = QPushButton("▶ Run Download + Train + Backtest")
        self.run_btn.clicked.connect(self.run_backtest)
        header.addWidget(self.run_btn)

        self.validate_btn = QPushButton("🔍 Run Full Validation")
        self.validate_btn.clicked.connect(self.run_validation)
        self.validate_btn.setToolTip("Run out-of-sample testing, walk-forward analysis, and risk metrics")
        header.addWidget(self.validate_btn)

        self.refresh_btn = QPushButton("Refresh Artifacts")
        self.refresh_btn.clicked.connect(self.load_artifacts)
        header.addWidget(self.refresh_btn)

        layout.addLayout(header)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Report tab
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.tabs.addTab(self.report_text, "Classification Report")

        # Confusion matrix tab
        self.cm_label = QLabel("No confusion matrix yet.")
        self.cm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cm_scroll = QScrollArea()
        self.cm_scroll.setWidgetResizable(True)
        self.cm_scroll.setWidget(self.cm_label)
        self.tabs.addTab(self.cm_scroll, "Confusion Matrix")

        # Backtest chart tab
        self.bt_label = QLabel("No backtest chart yet.")
        self.bt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bt_scroll = QScrollArea()
        self.bt_scroll.setWidgetResizable(True)
        self.bt_scroll.setWidget(self.bt_label)
        self.tabs.addTab(self.bt_scroll, "Backtest Chart")

        # Backtest CSV tab
        self.bt_table = QTableWidget()
        self.tabs.addTab(self.bt_table, "Backtest Results (CSV)")

        # Validation Report tab
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.tabs.addTab(self.validation_text, "Validation Report")

        # Out-of-Sample Comparison tab
        self.oos_label = QLabel("No out-of-sample comparison yet.")
        self.oos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.oos_scroll = QScrollArea()
        self.oos_scroll.setWidgetResizable(True)
        self.oos_scroll.setWidget(self.oos_label)
        self.tabs.addTab(self.oos_scroll, "Out-of-Sample Test")

        # Feature Importance tab
        self.fi_label = QLabel("No feature importance analysis yet.")
        self.fi_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fi_scroll = QScrollArea()
        self.fi_scroll.setWidgetResizable(True)
        self.fi_scroll.setWidget(self.fi_label)
        self.tabs.addTab(self.fi_scroll, "Feature Importance")

    def run_backtest(self):
        resp = self.ipc_client.send_command({"command": "run_backtest"}, timeout=5.0)
        if resp.get("success"):
            QMessageBox.information(self, "Backtest", "Backtest started. This can take a while.")
        else:
            msg = resp.get("message") or resp.get("error") or "Unknown error"
            QMessageBox.critical(self, "Backtest", f"Failed to start backtest: {msg}")
        self.refresh_status()

    def run_validation(self):
        """Run full validation in background and show results."""
        reply = QMessageBox.question(
            self,
            "Run Validation",
            "This will run comprehensive validation tests including:\n\n"
            "• Out-of-sample testing\n"
            "• Walk-forward analysis (may take 10-30 minutes)\n"
            "• Monte Carlo simulation\n"
            "• Feature importance analysis\n\n"
            "Do you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            return

        # Ask if they want to skip walk-forward (which takes longest)
        skip_wf = QMessageBox.question(
            self,
            "Walk-Forward Analysis",
            "Walk-forward analysis can take 10-30 minutes.\n\n"
            "Skip it for faster results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        try:
            # Run validation script in background
            cmd = [sys.executable, "run_full_validation.py", "--skip-download"]

            if skip_wf == QMessageBox.StandardButton.Yes:
                cmd.append("--skip-walkforward")

            logger.info(f"Running validation: {' '.join(cmd)}")

            # Run in subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            if result.returncode == 0:
                QMessageBox.information(
                    self,
                    "Validation Complete",
                    "Validation completed successfully!\n\n"
                    "Check the new tabs for results:\n"
                    "• Validation Report\n"
                    "• Out-of-Sample Test\n"
                    "• Feature Importance"
                )
                # Refresh artifacts to show new results
                self.load_artifacts()
            else:
                QMessageBox.warning(
                    self,
                    "Validation Failed",
                    f"Validation failed with errors.\n\n"
                    f"Check artifacts/validation.log for details.\n\n"
                    f"Error: {result.stderr[:500]}"
                )

        except subprocess.TimeoutExpired:
            QMessageBox.critical(
                self,
                "Timeout",
                "Validation timed out after 30 minutes.\n\n"
                "Try running with --skip-walkforward option."
            )
        except Exception as e:
            logger.error(f"Error running validation: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to run validation: {e}"
            )

    def refresh_status(self):
        resp = self.ipc_client.send_command({"command": "get_backtest_status"}, timeout=5.0)
        if resp.get("error"):
            self.status_label.setText(f"Status: error - {resp.get('error')}")
            return
        running = resp.get("running", False)
        last_run = resp.get("last_run") or "—"
        err = resp.get("error")
        phase = resp.get("phase") or "—"
        progress = resp.get("progress") or "—"
        started_at = resp.get("started_at") or "—"
        last_update = resp.get("last_update") or "—"
        thread_alive = resp.get("thread_alive", False)
        if running:
            self.status_label.setText(
                f"Status: running…  phase={phase}  progress={progress}  started_at={started_at}  last_update={last_update}"
            )
            self.run_btn.setEnabled(False)
        else:
            self.run_btn.setEnabled(True)
            if err:
                self.status_label.setText(
                    f"Status: failed (last_run={last_run})  phase={phase}  thread_alive={thread_alive} - {err}"
                )
            else:
                self.status_label.setText(f"Status: idle (last_run={last_run})  phase={phase}")

    def load_artifacts(self):
        # Report
        if self.report_path.exists():
            try:
                self.report_text.setPlainText(self.report_path.read_text())
            except Exception as e:
                self.report_text.setPlainText(f"Failed to read {self.report_path}: {e}")
        else:
            self.report_text.setPlainText("No classification_report.txt found yet.")

        # Confusion matrix image
        self._load_image(self.cm_path, self.cm_label, "No confusion_matrix.png found yet.")

        # Backtest image
        self._load_image(self.bt_png_path, self.bt_label, "No backtesting_results.png found yet.")

        # Backtest CSV
        if self.bt_csv_path.exists():
            try:
                df = pd.read_csv(self.bt_csv_path)
                self._populate_table(df)
            except Exception as e:
                logger.error(f"Failed to read {self.bt_csv_path}: {e}")
        else:
            self.bt_table.setRowCount(0)
            self.bt_table.setColumnCount(0)

        # Validation Report
        if self.validation_report_path.exists():
            try:
                self.validation_text.setPlainText(self.validation_report_path.read_text())
            except Exception as e:
                self.validation_text.setPlainText(f"Failed to read {self.validation_report_path}: {e}")
        else:
            self.validation_text.setPlainText("No validation_report.txt found yet.\n\nRun: ./run_validation.sh to generate validation report.")

        # Out-of-Sample Comparison
        self._load_image(self.oos_comparison_path, self.oos_label, "No out-of-sample comparison yet.\n\nRun: ./run_validation.sh to generate.")

        # Feature Importance
        self._load_image(self.feature_importance_path, self.fi_label, "No feature importance analysis yet.\n\nRun: ./run_validation.sh to generate.")

    def _load_image(self, path: Path, label: QLabel, missing_text: str):
        if not path.exists():
            label.setText(missing_text)
            label.setPixmap(QPixmap())
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            label.setText(f"Failed to load image: {path}")
            return
        label.setPixmap(pixmap)

    def _populate_table(self, df: pd.DataFrame):
        self.bt_table.setRowCount(len(df))
        self.bt_table.setColumnCount(len(df.columns))
        self.bt_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                val = df.iloc[r][col]
                self.bt_table.setItem(r, c, QTableWidgetItem(str(val)))
        self.bt_table.resizeColumnsToContents()

    def closeEvent(self, event):
        event.ignore()
        self.hide()
