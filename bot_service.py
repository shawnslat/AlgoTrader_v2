"""
Trading Bot Service - Non-blocking background service for GUI integration
Wraps the core trading logic from Trader_main_Grok4_20250731.py
"""
import os
import sys
import threading
import time
import logging
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
import shlex
import pytz
import schedule

# Logging (write to logs/bot_service.log for GUI troubleshooting)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/bot_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import core bot components
from Trader_main_Grok4_20250731 import (
    load_configuration,
    initialize_alpaca_api,
    fetch_current_market_data_with_crypto,
    fetch_latest_data,
    engineer_features,
    generate_signals,
    execute_trading_logic_live,
    fetch_current_positions,
    load_q_table,
    download_historical_data_with_crypto,
    load_historical_data,
    add_sentiment_features,
    prepare_train_test_data,
    tune_and_train_model,
    evaluate_model,
    backtest_strategy,
    maybe_override_tickers_from_json,
    maybe_fetch_tickers_via_dexter,
    maybe_add_crypto_tickers,
)
from pdt_guard import DayTradeGuard
from dexter_gate import DexterGate
from ipc_protocol import IPCServer, LogStreamer
from position_monitor import check_and_enforce_stops
import joblib

logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


class TradingBotService:
    """Background service that runs the trading bot and responds to GUI commands"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = None
        self.api = None
        self.model = None
        self.q_table = None
        self.selected_features = []
        self.guard = None
        self.dexter_gate = None

        # Runtime state
        self.running = False
        self.bot_thread = None
        self.raw_historical = None
        self.last_execution_time = None
        self.next_execution_time = None
        self.trade_count = 0
        self.current_signals = []
        self.last_signals_refresh = None
        self.signals_refresh_running = False
        self.signals_refresh_error = None
        self.backtest_running = False
        self.backtest_last_run = None
        self.backtest_last_error = None
        self.backtest_phase = "idle"
        self.backtest_started_at = None
        self.backtest_last_update = None
        self.backtest_progress = None
        self._backtest_thread = None
        self.status = "stopped"  # stopped, idle, running, trading, error
        self.error_message = None
        self.last_heartbeat = None
        self.dexter_last_answer = None
        self.dexter_last_query = None
        self.dexter_update_running = False
        self.dexter_update_error = None
        self.dexter_update_last = None
        self.dexter_bias_running = False
        self.dexter_bias_error = None
        self.dexter_bias_last = None

        # IPC
        self.ipc_server = None
        self.log_streamer = LogStreamer()

        # Thread safety
        self.state_lock = threading.RLock()
        self.job_lock = threading.Lock()
        self.heartbeat_thread = None

    def initialize(self):
        """Initialize all bot components"""
        try:
            logger.info("Initializing trading bot service...")

            # Load configuration
            self.config = load_configuration(self.config_path)
            # Do not block service startup on Dexter ticker research. We will kick it off in the background
            # after IPC is up so the GUI can connect quickly.
            self.config = maybe_override_tickers_from_json(self.config)
            self.config = maybe_add_crypto_tickers(self.config)

            # Initialize Alpaca API
            self.api = initialize_alpaca_api(
                self.config['alpaca']['api_key'],
                self.config['alpaca']['api_secret'],
                self.config['alpaca']['base_url']
            )

            # Initialize guards
            self.guard = DayTradeGuard(self.config.get('max_day_trades', 2))
            self.dexter_gate = DexterGate()

            # Load model if exists
            model_path = Path("artifacts/final_model.pkl")
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info("Loaded existing model from artifacts/final_model.pkl")
            else:
                logger.warning("No model found at artifacts/final_model.pkl - bot will need training")
                self.status = "error"
                self.error_message = "Model not found. Please train the model first."
                return False

            # Load Q-table
            self.q_table = load_q_table()  # uses artifacts/q_table.csv
            logger.info("Loaded/initialized Q-table from artifacts/q_table.csv")

            # Set selected features
            self.selected_features = [
                'MA10', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'Bollinger_Upper',
                'Bollinger_Lower', 'Lag1_Close', 'Lag2_Close', 'ATR', 'Stochastic_RSI', 'Volume_Change'
            ]

            # Add TA-Lib features if available (keep consistent with Trader_main module)
            try:
                import Trader_main_Grok4_20250731 as core_bot
                if getattr(core_bot, "talib", None):
                    self.selected_features.extend(['Momentum', 'SMA_20'])
                    logger.info("TA-Lib available - added Momentum and SMA_20 features")
                else:
                    logger.info("TA-Lib not available - using core features only")
            except Exception:
                logger.info("TA-Lib not available - using core features only")

            # Start IPC server
            self.ipc_server = IPCServer(self._handle_command)
            self.ipc_server.start()

            self._start_heartbeat()
            self._start_dexter_autofetch()
            self.status = "idle"
            logger.info("Trading bot service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing bot service: {e}", exc_info=True)
            self.status = "error"
            self.error_message = str(e)
            return False

    def _handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming commands from GUI"""
        cmd = command.get('command')

        try:
            if cmd == 'start':
                return self.start_trading()
            elif cmd == 'stop':
                return self.stop_trading()
            elif cmd == 'run_now':
                return self.run_now()
            elif cmd == 'get_status':
                return self.get_status()
            elif cmd == 'get_positions':
                return self.get_positions()
            elif cmd == 'get_signals':
                return self.get_signals()
            elif cmd == 'refresh_signals':
                return self.refresh_signals()
            elif cmd == 'get_signals_status':
                return self.get_signals_status()
            elif cmd == 'get_account':
                return self.get_account_info()
            elif cmd == 'run_backtest':
                return self.run_backtest()
            elif cmd == 'get_backtest_status':
                return self.get_backtest_status()
            elif cmd == 'dexter_chat':
                return self.dexter_chat(command.get('query', ''), command.get('include_context', False))
            elif cmd == 'dexter_update_tickers':
                return self.dexter_update_tickers()
            elif cmd == 'dexter_generate_bias':
                return self.dexter_generate_bias()
            elif cmd == 'get_dexter_status':
                return self.get_dexter_status()
            elif cmd == 'manual_trade':
                return self.execute_manual_trade(
                    command.get('ticker'),
                    command.get('action'),
                    command.get('quantity')
                )
            elif cmd == 'refresh_dexter':
                self.dexter_gate.refresh()
                return {"success": True, "message": "Dexter bias refreshed"}
            else:
                return {"error": f"Unknown command: {cmd}"}

        except Exception as e:
            logger.error(f"Error handling command {cmd}: {e}", exc_info=True)
            return {"error": str(e)}

    def dexter_chat(self, query: str, include_context: bool = False) -> Dict[str, Any]:
        """
        Run a Dexter "chat" query by calling a configured shell command.

        Configure either:
        - env: DEXTER_CHAT_CMD
        - config.yaml: dexter_chat_command

        The command can accept the query via:
        - environment variable DEXTER_QUERY
        - or as a final shell-quoted argument appended by this function.

        Return format (expected): JSON {"answer": "..."} on stdout. If stdout isn't JSON, it is returned as plain text.
        """
        if not isinstance(query, str) or not query.strip():
            return {"error": "Missing query."}

        cmd = os.getenv("DEXTER_CHAT_CMD") or (self.config or {}).get("dexter_chat_command")
        if not cmd:
            return {"error": "Dexter chat not configured. Set DEXTER_CHAT_CMD or config.yaml dexter_chat_command."}

        timeout_s = int((self.config or {}).get("dexter_chat_timeout_seconds", 120))
        full_query = query.strip()

        # Optional context injection (lightweight)
        if include_context:
            try:
                acct = self.get_account_info()
                pos = self.get_positions()
                sig = self.get_signals()
                context = {
                    "account": acct if not acct.get("error") else None,
                    "positions": pos.get("positions") if not pos.get("error") else None,
                    "signals": sig.get("signals", []),
                }
                full_query = (
                    "Context (JSON):\n"
                    + str(context)
                    + "\n\nUser question:\n"
                    + full_query
                )
            except Exception:
                pass

        logger.info("Dexter chat request received.")
        env = os.environ.copy()
        env["DEXTER_QUERY"] = full_query

        # Append query as an argument too (helps if the command expects argv).
        cmd_with_arg = f"{cmd} {shlex.quote(full_query)}"
        try:
            result = subprocess.run(cmd_with_arg, shell=True, capture_output=True, text=True, timeout=timeout_s, env=env)
            if result.returncode != 0:
                return {"error": f"Dexter command failed (exit {result.returncode}): {result.stderr.strip() or 'unknown error'}"}
            stdout = (result.stdout or "").strip()
            if not stdout:
                return {"error": "Dexter returned empty output."}
            try:
                import json
                data = json.loads(stdout)
                if isinstance(data, dict) and data.get("error"):
                    return {"error": str(data.get("error"))}
                answer = data.get("answer") if isinstance(data, dict) else None
                if not answer:
                    answer = stdout
            except Exception:
                answer = stdout

            with self.state_lock:
                self.dexter_last_query = query
                self.dexter_last_answer = answer

            return {"success": True, "answer": answer}
        except Exception as e:
            logger.error(f"Dexter chat failed: {e}", exc_info=True)
            return {"error": str(e)}

    def get_dexter_status(self) -> Dict[str, Any]:
        with self.state_lock:
            return {
                "tickers_update_running": self.dexter_update_running,
                "tickers_update_last": self.dexter_update_last,
                "tickers_update_error": self.dexter_update_error,
                "bias_running": self.dexter_bias_running,
                "bias_last": self.dexter_bias_last,
                "bias_error": self.dexter_bias_error,
            }

    def dexter_update_tickers(self) -> Dict[str, Any]:
        """Explicitly refresh tickers_auto.json via Dexter command (background)."""
        with self.state_lock:
            if self.dexter_update_running:
                return {"success": False, "message": "Dexter ticker update already running."}
            self.dexter_update_running = True
            self.dexter_update_error = None
        threading.Thread(target=self._dexter_update_tickers_job, daemon=True).start()
        return {"success": True, "message": "Dexter ticker update started."}

    def _dexter_update_tickers_job(self):
        try:
            self._dexter_autofetch_job()
            with self.state_lock:
                self.dexter_update_last = datetime.now().isoformat(timespec="seconds")
        except Exception as e:
            with self.state_lock:
                self.dexter_update_error = str(e)
        finally:
            with self.state_lock:
                self.dexter_update_running = False

    def dexter_generate_bias(self) -> Dict[str, Any]:
        """Generate dexter_bias.json via a configured Dexter bias command (background)."""
        with self.state_lock:
            if self.dexter_bias_running:
                return {"success": False, "message": "Dexter bias generation already running."}
            self.dexter_bias_running = True
            self.dexter_bias_error = None
        threading.Thread(target=self._dexter_bias_job, daemon=True).start()
        return {"success": True, "message": "Dexter bias generation started."}

    def _dexter_bias_job(self):
        try:
            cmd = os.getenv("DEXTER_BIAS_CMD") or (self.config or {}).get("dexter_bias_command")
            if not cmd:
                raise RuntimeError("Dexter bias not configured. Set DEXTER_BIAS_CMD or config.yaml dexter_bias_command.")

            timeout_s = int((self.config or {}).get("dexter_bias_timeout_seconds", 90))
            tickers = (self.config or {}).get("tickers") or []
            env = os.environ.copy()
            env["DEXTER_TICKERS"] = ",".join([str(t).upper() for t in tickers])
            logger.info(f"Running Dexter bias command (timeout={timeout_s}s)...")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_s, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"Dexter bias command failed (exit {result.returncode}): {result.stderr.strip() or 'unknown error'}")

            stdout = (result.stdout or "").strip()
            if not stdout:
                raise RuntimeError("Dexter bias command returned empty output.")

            import json
            data = json.loads(stdout)
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(str(data.get("error")))

            # Expect dict mapping tickers to decision objects OR {"bias": {...}}
            bias = data.get("bias") if isinstance(data, dict) and isinstance(data.get("bias"), dict) else data
            if not isinstance(bias, dict) or not bias:
                raise RuntimeError("Dexter bias output missing bias object.")

            with open("dexter_bias.json", "w") as f:
                json.dump(bias, f, indent=2)

            # Enrich with FinViz fundamental data
            try:
                from finviz_enrichment import enrich_dexter_bias, FINVIZ_AVAILABLE
                if FINVIZ_AVAILABLE:
                    logger.info("Enriching bias with FinViz fundamental data...")
                    enrich_dexter_bias("dexter_bias.json")
                    logger.info("FinViz enrichment complete.")
            except Exception as finviz_err:
                logger.warning(f"FinViz enrichment failed (non-fatal): {finviz_err}")

            # Refresh gate so live trading uses the new rules immediately
            try:
                self.dexter_gate.refresh()
            except Exception:
                pass

            with self.state_lock:
                self.dexter_bias_last = datetime.now().isoformat(timespec="seconds")
        except Exception as e:
            logger.error(f"Dexter bias generation failed: {e}", exc_info=True)
            with self.state_lock:
                self.dexter_bias_error = str(e)
        finally:
            with self.state_lock:
                self.dexter_bias_running = False

    def _start_dexter_autofetch(self):
        """Optionally run Dexter ticker research in the background."""
        try:
            if not (self.config or {}).get("dexter_autofetch"):
                return
            t = threading.Thread(target=self._dexter_autofetch_job, daemon=True)
            t.start()
        except Exception as e:
            logger.error(f"Failed to start Dexter autofetch thread: {e}")

    def _dexter_autofetch_job(self):
        """
        Runs dexter_ticker_command (or env DEXTER_TICKER_CMD) and writes tickers_auto.json.
        This is best-effort and must never prevent the service from running.
        """
        try:
            cmd = os.getenv("DEXTER_TICKER_CMD") or (self.config or {}).get("dexter_ticker_command")
            if not cmd:
                logger.info("Dexter autofetch enabled but no dexter_ticker_command/DEXTER_TICKER_CMD set.")
                return
            timeout_s = int((self.config or {}).get("dexter_ticker_timeout_seconds", 45))
            logger.info(f"Starting Dexter ticker autofetch (timeout={timeout_s}s)...")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_s, env=os.environ.copy())
            if result.returncode != 0:
                logger.error(f"Dexter ticker command failed (exit {result.returncode}): {result.stderr.strip() or 'unknown error'}")
                return
            stdout = (result.stdout or "").strip()
            if not stdout:
                logger.error("Dexter ticker command returned empty output.")
                return
            import json
            data = json.loads(stdout)
            if isinstance(data, dict) and data.get("error"):
                logger.error(f"Dexter ticker command returned error: {data.get('error')}")
                return
            tickers = data.get("tickers") if isinstance(data, dict) else None
            if not isinstance(tickers, list) or not tickers:
                logger.error("Dexter ticker command returned no tickers.")
                return
            cleaned = []
            for t in tickers:
                if isinstance(t, str) and t.strip():
                    cleaned.append(t.strip().upper())
            cleaned = list(dict.fromkeys(cleaned))
            if not cleaned:
                logger.error("Dexter ticker command returned no valid tickers after cleaning.")
                return

            with open("tickers_auto.json", "w") as f:
                json.dump({"tickers": cleaned}, f)
            logger.info(f"Dexter tickers updated: {cleaned}")

            # Reload config to apply overrides for future cycles/signals
            self.config = maybe_override_tickers_from_json(self.config)
            self.config = maybe_add_crypto_tickers(self.config)
        except subprocess.TimeoutExpired:
            logger.error("Dexter ticker autofetch timed out.")
        except Exception as e:
            logger.error(f"Dexter ticker autofetch failed: {e}", exc_info=True)

    def start_trading(self) -> Dict[str, Any]:
        """Start the trading bot"""
        with self.state_lock:
            if self.running:
                return {"success": False, "message": "Bot is already running"}

            if self.status == "error":
                return {"success": False, "message": f"Cannot start: {self.error_message}"}

            self.running = True
            self._set_status("running")
            self.bot_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.bot_thread.start()

            logger.info("Trading bot started")
            return {"success": True, "message": "Bot started successfully"}

    def stop_trading(self) -> Dict[str, Any]:
        """Stop the trading bot"""
        with self.state_lock:
            if not self.running:
                return {"success": False, "message": "Bot is not running"}

            self.running = False
            self._set_status("stopped")
            logger.info("Trading bot stopped")
            return {"success": True, "message": "Bot stopped successfully"}

    def run_now(self) -> Dict[str, Any]:
        """Trigger an immediate trading cycle (useful for testing)."""
        with self.state_lock:
            if not self.running:
                return {"success": False, "message": "Bot is not running. Click Start Bot first."}
        threading.Thread(target=self._execute_cycle, kwargs={"force": True}, daemon=True).start()
        return {"success": True, "message": "Triggered run-now"}

    def _start_heartbeat(self):
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    def _heartbeat_loop(self):
        while True:
            self.last_heartbeat = datetime.now().isoformat(timespec="seconds")
            time.sleep(1)

    def _execute_cycle(self, force: bool = False, session: str = 'afternoon'):
        """Execute a single trading cycle, optionally bypassing the time window.

        Args:
            force: If True, bypass time window checks
            session: 'morning' for gap trading, 'afternoon' for EOD signals
        """
        with self.job_lock:
            current_time = datetime.now(tz=pytz.timezone('US/Eastern'))
            logger.info(f"Cycle running at {current_time.strftime('%H:%M:%S')} EDT (force={force}, session={session})")

            now_time = current_time.time()

            # Define execution windows
            morning_start = dt_time(10, 0)
            morning_end = dt_time(10, 15)
            afternoon_start = dt_time(15, 15)
            afternoon_end = dt_time(15, 30)

            if not force:
                # Check if we're in a valid execution window
                in_morning_window = morning_start <= now_time <= morning_end
                in_afternoon_window = afternoon_start <= now_time <= afternoon_end
                is_weekday = current_time.weekday() < 5

                if not is_weekday:
                    logger.info("Weekend - skipping cycle.")
                    self._set_status("running")
                    return

                if session == 'morning' and not in_morning_window:
                    logger.info("Outside morning execution window (10:00-10:15 AM ET). Skipping cycle.")
                    self._set_status("running")
                    return
                elif session == 'afternoon' and not in_afternoon_window:
                    logger.info("Outside afternoon execution window (3:15-3:30 PM ET). Skipping cycle.")
                    self._set_status("running")
                    return

            self._set_status("trading")
            try:
                # FIRST: Check existing positions for stop loss enforcement
                # This catches positions that should have been stopped out
                logger.info("Checking existing positions for stop loss enforcement...")
                stop_actions = check_and_enforce_stops(self.api, self.config)
                if stop_actions:
                    logger.warning(f"Position monitor closed {len(stop_actions)} positions")

                # Fetch historical data if needed
                if self.raw_historical is None:
                    self.raw_historical = fetch_current_market_data_with_crypto(
                        self.config['tickers'],
                        self.config['polygon']['api_key'],
                        self.api,
                        config=self.config,
                    )
                    if self.raw_historical.empty:
                        raise RuntimeError("Failed to fetch historical data")

                # Fetch latest data
                latest_data = fetch_latest_data(self.config['tickers'], self.api, config=self.config)
                if latest_data.empty:
                    logger.warning("No new data fetched. Skipping this cycle.")
                    self._set_status("running")
                    return

                import pandas as pd
                self.raw_historical = pd.concat([self.raw_historical, latest_data]).drop_duplicates(
                    subset=['ticker', 'date'], keep='last'
                )
                self.raw_historical = self.raw_historical.groupby('ticker').tail(200).reset_index(drop=True)

                engineered_data = engineer_features(self.raw_historical)
                if engineered_data.empty:
                    logger.warning("Feature engineering resulted in empty DataFrame. Skipping this cycle.")
                    self._set_status("running")
                    return
                try:
                    engineered_data = add_sentiment_features(engineered_data, self.config)
                except Exception as e:
                    logger.error(f"Sentiment analysis failed: {e}. Using zero sentiment as fallback.")
                    engineered_data['Sentiment_Score'] = 0.0

                positions_snapshot = {p['ticker']: p for p in fetch_current_positions(self.api)}

                signals_df = generate_signals(
                    self.model,
                    engineered_data,
                    self.selected_features + ['Sentiment_Score'],
                    self.config,
                    self.q_table,
                    positions_snapshot=positions_snapshot,
                )

                if signals_df.empty:
                    with self.state_lock:
                        self.current_signals = []
                    self._set_status("running")
                    return

                latest_signals = signals_df.sort_values(['ticker', 'date']).groupby('ticker').tail(1)
                with self.state_lock:
                    self.current_signals = self._format_signals_for_gui(latest_signals)

                actionable = latest_signals[latest_signals['Signal'].isin([1, -1])]
                if not actionable.empty:
                    execute_trading_logic_live(
                        self.api,
                        actionable,
                        self.config,
                        self.q_table,
                        self.guard,
                        self.dexter_gate,
                        buying_power_pct=float((self.config or {}).get("buying_power_pct", 100)),
                    )
                    with self.state_lock:
                        self.trade_count += int(len(actionable))

                with self.state_lock:
                    self.last_execution_time = current_time
                self._set_status("running")
            except Exception as e:
                logger.error(f"Error in cycle: {e}", exc_info=True)
                self._set_status("error", error=str(e))

    def _set_status(self, status: str, error: str | None = None):
        with self.state_lock:
            self.status = status
            if error is not None:
                self.error_message = error

    def _trading_loop(self):
        """Main trading loop - runs in background thread"""
        # Get schedule config (default to 2x daily: morning gaps + afternoon close)
        schedule_config = (self.config or {}).get('schedule', {})
        morning_time = schedule_config.get('morning_run', '10:00')  # 30 min after open for gap confirmation
        afternoon_time = schedule_config.get('afternoon_run', '15:15')  # 45 min before close
        enable_morning = schedule_config.get('enable_morning_run', True)

        if enable_morning:
            logger.info(f"Starting 2x daily trading mode (Morning: {morning_time} ET, Afternoon: {afternoon_time} ET)...")
        else:
            logger.info(f"Starting daily close trading mode (executes at {afternoon_time} ET)...")

        def morning_job():
            """Morning job - catch overnight gaps"""
            logger.info("=== MORNING RUN: Checking for gap opportunities ===")
            self._execute_cycle(force=False, session='morning')

        def afternoon_job():
            """Afternoon job - end of day signals"""
            logger.info("=== AFTERNOON RUN: End of day trading signals ===")
            self._execute_cycle(force=False, session='afternoon')

        # Check if we should run immediately based on current time
        current_time = datetime.now(tz=pytz.timezone('US/Eastern'))
        now_time = current_time.time()

        # Morning window: 10:00-10:15 AM ET
        morning_start = dt_time(10, 0)
        morning_end = dt_time(10, 15)

        # Afternoon window: 3:15-3:30 PM ET
        afternoon_start = dt_time(15, 15)
        afternoon_end = dt_time(15, 30)

        if current_time.weekday() < 5:  # Weekday
            if enable_morning and morning_start <= now_time <= morning_end:
                logger.info("Within morning execution window at startup. Running morning job...")
                self._execute_cycle(force=False, session='morning')
            elif afternoon_start <= now_time <= afternoon_end:
                logger.info("Within afternoon execution window at startup. Running afternoon job...")
                self._execute_cycle(force=False, session='afternoon')
            else:
                logger.info(f"Outside execution windows at startup. Next runs: {morning_time} AM, {afternoon_time} PM ET")

        # Schedule morning run (gap trading)
        if enable_morning:
            schedule.every().day.at(morning_time).do(morning_job)
            logger.info(f"Scheduled morning execution at {morning_time} ET (gap trading)")

        # Schedule afternoon run (end of day)
        schedule.every().day.at(afternoon_time).do(afternoon_job)
        logger.info(f"Scheduled afternoon execution at {afternoon_time} ET (EOD signals)")

        # Calculate next execution time
        self._update_next_execution_time()

        # Main loop
        while self.running:
            schedule.run_pending()
            self._update_next_execution_time()
            time.sleep(60)  # Check every minute

        logger.info("Trading loop stopped")

    def _update_next_execution_time(self):
        """Update the next scheduled execution time"""
        current_time = datetime.now(tz=pytz.timezone('US/Eastern'))
        next_run = schedule.next_run()

        if next_run:
            self.next_execution_time = next_run.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Calculate tomorrow's first execution time (morning run)
            tomorrow = current_time + timedelta(days=1)
            schedule_config = (self.config or {}).get('schedule', {})
            morning_time = schedule_config.get('morning_run', '10:00')
            hour, minute = map(int, morning_time.split(':'))
            self.next_execution_time = tomorrow.replace(hour=hour, minute=minute, second=0).strftime("%Y-%m-%d %H:%M:%S")

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        with self.state_lock:
            return {
                "status": self.status,
                "running": self.running,
                "next_execution": self.next_execution_time,
                "last_execution": self.last_execution_time.strftime("%Y-%m-%d %H:%M:%S") if self.last_execution_time else None,
                "trade_count": self.trade_count,
                "last_heartbeat": self.last_heartbeat,
                "error": self.error_message
            }

    def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        try:
            raw_positions = fetch_current_positions(self.api)
            positions = []
            for p in raw_positions:
                qty = float(p.get('quantity', 0))
                avg_entry = float(p.get('average_price', 0))
                current = float(p.get('current_price', avg_entry))
                market_value = qty * current
                unrealized_pl = (current - avg_entry) * qty if avg_entry else 0.0
                unrealized_plpc = ((current - avg_entry) / avg_entry) * 100 if avg_entry else 0.0
                positions.append({
                    "ticker": p.get('ticker'),
                    "qty": qty,
                    "avg_entry_price": avg_entry,
                    "current_price": current,
                    "market_value": market_value,
                    "unrealized_pl": unrealized_pl,
                    "unrealized_plpc": unrealized_plpc,
                })
            return {"positions": positions}
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {"error": str(e)}

    def get_signals(self) -> Dict[str, Any]:
        """Get current trading signals"""
        with self.state_lock:
            return {"signals": self.current_signals}

    def refresh_signals(self) -> Dict[str, Any]:
        """Kick off a background signals refresh (non-blocking for GUI)."""
        with self.state_lock:
            if self.signals_refresh_running:
                return {"success": False, "message": "Signals refresh already running."}
            self.signals_refresh_running = True
            self.signals_refresh_error = None
        threading.Thread(target=self._signals_refresh_job, daemon=True).start()
        return {"success": True, "message": "Signals refresh started."}

    def get_signals_status(self) -> Dict[str, Any]:
        with self.state_lock:
            return {
                "running": self.signals_refresh_running,
                "last_refresh": self.last_signals_refresh.isoformat(timespec="seconds") if self.last_signals_refresh else None,
                "error": self.signals_refresh_error,
            }

    def _signals_refresh_job(self):
        """Compute latest signals without executing trades."""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Train first.")

            # Refresh config/tickers (Dexter/JSON) in case they changed
            self.config = load_configuration(self.config_path)
            maybe_fetch_tickers_via_dexter(self.config)
            self.config = maybe_override_tickers_from_json(self.config)
            self.config = maybe_add_crypto_tickers(self.config)

            raw = fetch_current_market_data_with_crypto(
                self.config['tickers'],
                self.config['polygon']['api_key'],
                self.api,
                config=self.config,
            )
            if raw.empty:
                raise RuntimeError("No market data returned.")

            latest = fetch_latest_data(self.config['tickers'], self.api, config=self.config)
            if not latest.empty:
                import pandas as pd
                raw = pd.concat([raw, latest]).drop_duplicates(subset=['ticker', 'date'], keep='last')

            engineered = engineer_features(raw)
            if engineered.empty:
                raise RuntimeError("Feature engineering produced empty data.")
            engineered = add_sentiment_features(engineered, self.config)

            positions_snapshot = {p['ticker']: p for p in fetch_current_positions(self.api)}
            signals_df = generate_signals(
                self.model,
                engineered,
                self.selected_features + ['Sentiment_Score'],
                self.config,
                self.q_table,
                positions_snapshot=positions_snapshot,
            )
            if signals_df.empty:
                with self.state_lock:
                    self.current_signals = []
                    self.last_signals_refresh = datetime.now()
                return

            latest_signals = signals_df.sort_values(['ticker', 'date']).groupby('ticker').tail(1)
            formatted = self._format_signals_for_gui(latest_signals)
            with self.state_lock:
                self.current_signals = formatted
                self.last_signals_refresh = datetime.now()
        except Exception as e:
            logger.error(f"Error refreshing signals: {e}", exc_info=True)
            with self.state_lock:
                self.signals_refresh_error = str(e)
        finally:
            with self.state_lock:
                self.signals_refresh_running = False

    def run_backtest(self) -> Dict[str, Any]:
        """Start a background job to download/train/backtest."""
        with self.state_lock:
            if self.backtest_running:
                return {"success": False, "message": "Backtest already running."}
            self.backtest_running = True
            self.backtest_last_error = None
            self.backtest_started_at = datetime.now().isoformat(timespec="seconds")
            self.backtest_last_update = self.backtest_started_at
            self.backtest_phase = "starting"
            self.backtest_progress = "Queued"
        self._backtest_thread = threading.Thread(target=self._backtest_job, daemon=True)
        self._backtest_thread.start()
        return {"success": True, "message": "Backtest started."}

    def get_backtest_status(self) -> Dict[str, Any]:
        with self.state_lock:
            thread_alive = bool(self._backtest_thread and self._backtest_thread.is_alive())
            return {
                "running": self.backtest_running,
                "last_run": self.backtest_last_run,
                "error": self.backtest_last_error,
                "phase": self.backtest_phase,
                "started_at": self.backtest_started_at,
                "last_update": self.backtest_last_update,
                "progress": self.backtest_progress,
                "thread_alive": thread_alive,
            }

    def _backtest_job(self):
        try:
            logger.info("Starting GUI-triggered backtest job...")
            self._set_backtest_progress("loading_config", "Loading configuration and tickers")
            # Reload config/tickers (Dexter/JSON)
            self.config = load_configuration(self.config_path)
            maybe_fetch_tickers_via_dexter(self.config)
            self.config = maybe_override_tickers_from_json(self.config)
            self.config = maybe_add_crypto_tickers(self.config)

            tickers = self.config['tickers']
            os.makedirs("./data", exist_ok=True)
            self._set_backtest_progress("downloading", f"Downloading historical data ({len(tickers)} tickers)")
            download_historical_data_with_crypto(tickers, self.config)
            self._set_backtest_progress("loading_data", "Loading historical CSVs")
            data = load_historical_data("./data", self.config)
            if data.empty:
                raise RuntimeError("No data loaded from ./data")
            self._set_backtest_progress("feature_engineering", "Engineering features")
            data = engineer_features(data)
            self._set_backtest_progress("sentiment", "Adding sentiment features (can be slow)")
            data = add_sentiment_features(data, self.config)
            if data.empty:
                raise RuntimeError("No data after feature engineering/sentiment.")

            selected = list(self.selected_features) + ['Sentiment_Score']
            self._set_backtest_progress("train_split", "Preparing train/test data")
            X_train, X_test, y_train, y_test = prepare_train_test_data(data, selected)
            self._set_backtest_progress("training", "Training model (RandomizedSearchCV)")
            model = tune_and_train_model(X_train, y_train)
            if model is None:
                raise RuntimeError("Model training returned None.")
            self._set_backtest_progress("evaluation", "Evaluating model and saving artifacts")
            evaluate_model(model, X_test, y_test)
            with self.state_lock:
                self.model = model
            self._set_backtest_progress("backtesting", "Running backtest simulation")
            backtest_strategy(model, data, selected, self.config, self.q_table)

            with self.state_lock:
                self.backtest_last_run = datetime.now().isoformat(timespec="seconds")
                self.backtest_phase = "completed"
                self.backtest_progress = "Done"
                self.backtest_last_update = datetime.now().isoformat(timespec="seconds")
            logger.info("Backtest job completed successfully.")
        except Exception as e:
            logger.error(f"Backtest job failed: {e}", exc_info=True)
            with self.state_lock:
                self.backtest_last_error = str(e)
                self.backtest_phase = "failed"
                self.backtest_progress = "Failed"
                self.backtest_last_update = datetime.now().isoformat(timespec="seconds")
        finally:
            with self.state_lock:
                self.backtest_running = False

    def _set_backtest_progress(self, phase: str, message: str):
        now_s = datetime.now().isoformat(timespec="seconds")
        with self.state_lock:
            self.backtest_phase = phase
            self.backtest_progress = message
            self.backtest_last_update = now_s
        logger.info(f"[backtest] {phase}: {message}")

    def _format_signals_for_gui(self, df) -> List[Dict[str, Any]]:
        """Format latest per-ticker signals for the GUI table."""
        out: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            sig = int(row.get('Signal', 0))
            if sig == 1:
                action = "buy"
            elif sig == -1:
                action = "sell"
            else:
                action = "hold"
            out.append({
                "ticker": row.get('ticker', ''),
                "action": action,
                "confidence": float(row.get('Prediction', 0)),
                "timestamp": str(row.get('date', '')),
            })
        return out

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "pattern_day_trader": account.pattern_day_trader
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {"error": str(e)}

    def execute_manual_trade(self, ticker: str, action: str, quantity: int) -> Dict[str, Any]:
        """Execute a manual trade"""
        try:
            logger.info(f"Manual trade request: {action} {quantity} shares of {ticker}")

            # Basic validation
            if action not in ['buy', 'sell']:
                return {"error": "Action must be 'buy' or 'sell'"}

            if quantity <= 0:
                return {"error": "Quantity must be positive"}

            # Check with Dexter gate
            if not self.dexter_gate.should_allow(ticker, 999, {}):
                return {"error": f"Trade blocked by Dexter gate for {ticker}"}

            # Execute trade via Alpaca
            if action == 'buy':
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
            else:
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )

            logger.info(f"Manual trade executed: {order.id}")
            return {
                "success": True,
                "order_id": order.id,
                "message": f"{action.upper()} {quantity} {ticker}"
            }

        except Exception as e:
            logger.error(f"Error executing manual trade: {e}", exc_info=True)
            return {"error": str(e)}

    def shutdown(self):
        """Shutdown the bot service"""
        logger.info("Shutting down bot service...")
        self.stop_trading()
        if self.ipc_server:
            self.ipc_server.stop()
        logger.info("Bot service shutdown complete")


def main():
    """Run the bot service as a standalone process"""
    service = TradingBotService()

    if not service.initialize():
        logger.error("Failed to initialize bot service")
        sys.exit(1)

    # Auto-start trading
    service.start_trading()

    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        service.shutdown()


if __name__ == "__main__":
    main()
