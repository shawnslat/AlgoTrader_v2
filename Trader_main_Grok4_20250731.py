import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import matplotlib
# Use a non-interactive backend so plotting works from background threads (e.g., GUI-triggered backtests on macOS).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import csv
import yaml
import logging
import json
import subprocess
from datetime import datetime, timedelta, time as dt_time
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import requests
import ta
import pytz
import schedule
import asyncio
import aiohttp
import threading
from itertools import product
from pdt_guard import DayTradeGuard
from dexter_gate import DexterGate
from signal_confidence import calculate_confidence, get_aggregated_sentiment, ConfidenceScore, SignalDirection
try:
    import talib
except ImportError:
    talib = None
    logging.warning("TA-Lib is not installed. Technical indicators will not work.")
from openai import OpenAI
import tweepy

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_DIR = "logs"
ARTIFACT_DIR = "artifacts"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'master_trading_bot.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration and API Initialization
# =============================================================================

def load_configuration(config_file: str = "config.yaml") -> dict:
    """Load configuration from a YAML file."""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}", exc_info=True)
        raise

def maybe_override_tickers_from_json(config: dict, json_path: str = "tickers_auto.json") -> dict:
    """
    If tickers_auto.json exists, override tickers in config at runtime.
    Does not persist changes to config.yaml; keeps it purely in-memory.
    """
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            tickers = data.get("tickers") or []
            if isinstance(tickers, list):
                cleaned = []
                for t in tickers:
                    if isinstance(t, str) and t.strip():
                        cleaned.append(t.strip().upper())
                if cleaned:
                    config['tickers'] = list(dict.fromkeys(cleaned))
                    logger.info(f"Tickers overridden from {json_path}: {config['tickers']}")
                else:
                    logger.warning(f"{json_path} present but no valid tickers found; using config.yaml tickers.")
            else:
                logger.warning(f"{json_path} invalid format; expected list under 'tickers'. Using config.yaml tickers.")
        else:
            logger.info("tickers_auto.json not found; using tickers from config.yaml.")
    except Exception as e:
        logger.error(f"Failed to override tickers from {json_path}: {e}")
    return config

def maybe_fetch_tickers_via_dexter(config: dict, output_path: str = "tickers_auto.json"):
    """
    Optional: call an external Dexter command to generate tickers_auto.json.
    Controlled by config.dexter_autofetch and config.dexter_ticker_command or env DEXTER_TICKER_CMD.
    The command should print JSON with a top-level 'tickers' list to stdout.
    """
    try:
        if not config.get('dexter_autofetch'):
            return
        cmd = os.getenv("DEXTER_TICKER_CMD") or config.get('dexter_ticker_command')
        if not cmd:
            logger.warning("dexter_autofetch enabled but no dexter_ticker_command or DEXTER_TICKER_CMD set.")
            return
        logger.info(f"Running Dexter ticker command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error(f"Dexter ticker command failed (exit {result.returncode}): {result.stderr}")
            return
        data = json.loads(result.stdout)
        tickers = data.get("tickers") or []
        if not isinstance(tickers, list) or not tickers:
            logger.warning("Dexter ticker JSON missing/empty 'tickers' list.")
            return
        cleaned = []
        for t in tickers:
            if isinstance(t, str) and t.strip():
                cleaned.append(t.strip().upper())
        cleaned = list(dict.fromkeys(cleaned))
        if not cleaned:
            logger.warning("Dexter ticker command returned no valid tickers after cleaning.")
            return
        with open(output_path, 'w') as f:
            json.dump({"tickers": cleaned}, f)
        logger.info(f"Dexter tickers written to {output_path}: {cleaned}")
    except Exception as e:
        logger.error(f"Failed to fetch tickers via Dexter: {e}")

def initialize_alpaca_api(api_key: str, api_secret: str, base_url: str) -> tradeapi.REST:
    """Initialize the Alpaca API instance."""
    try:
        api = tradeapi.REST(api_key, api_secret, base_url)
        logger.info("Alpaca API initialized successfully.")
        return api
    except Exception as e:
        logger.error(f"Error initializing Alpaca API: {e}", exc_info=True)
        raise

# =============================================================================
# Data Fetching with Polygon.io and Alpaca
# =============================================================================

def _get_crypto_bars(api: tradeapi.REST, ticker: str, start_date: str, end_date: str):
    """Fetch crypto bars using Alpaca data API."""
    timeframe = tradeapi.TimeFrame(1, tradeapi.TimeFrameUnit.Day)
    try:
        if hasattr(api, "get_crypto_bars"):
            return api.get_crypto_bars(ticker, timeframe, start=start_date, end=end_date)
        return api.get_bars(ticker, timeframe, start=start_date, end=end_date)
    except Exception as e:
        logger.error(f"Error fetching crypto bars for {ticker}: {e}")
        return []

def _bars_to_rows(ticker: str, bars) -> List[dict]:
    """Normalize Alpaca bar objects into dict rows."""
    rows = []
    try:
        for bar in list(bars):
            bar_time = getattr(bar, "t", None)
            if bar_time is None:
                continue
            if isinstance(bar_time, (int, float)):
                dt = datetime.fromtimestamp(bar_time)
            elif hasattr(bar_time, "to_pydatetime"):
                dt = bar_time.to_pydatetime()
            else:
                dt = bar_time
            rows.append({
                'ticker': ticker,
                'date': dt.strftime('%Y-%m-%d'),
                'open': getattr(bar, "o", None),
                'high': getattr(bar, "h", None),
                'low': getattr(bar, "l", None),
                'close': getattr(bar, "c", None),
                'volume': getattr(bar, "v", None),
            })
    except Exception as e:
        logger.error(f"Failed to normalize bars for {ticker}: {e}")
    return rows

def _bars_to_csv_rows(bars) -> List[dict]:
    """Normalize Alpaca bars into CSV rows matching Polygon format."""
    rows = []
    try:
        for bar in list(bars):
            bar_time = getattr(bar, "t", None)
            if bar_time is None:
                continue
            if isinstance(bar_time, (int, float)):
                dt = datetime.fromtimestamp(bar_time)
            elif hasattr(bar_time, "to_pydatetime"):
                dt = bar_time.to_pydatetime()
            else:
                dt = bar_time
            rows.append({
                'time': dt.strftime('%Y-%m-%d'),
                'open': getattr(bar, "o", None),
                'high': getattr(bar, "h", None),
                'low': getattr(bar, "l", None),
                'close': getattr(bar, "c", None),
                'volume': getattr(bar, "v", None),
            })
    except Exception as e:
        logger.error(f"Failed to normalize bars for CSV: {e}")
    return rows

def download_historical_crypto_data(tickers: List[str], alpaca_cfg: dict, start_date: str, end_date: Optional[str] = None):
    """Download historical crypto data from Alpaca and save to CSV."""
    if not tickers:
        return
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    api_key = alpaca_cfg.get('api_key')
    api_secret = alpaca_cfg.get('api_secret')
    base_url = alpaca_cfg.get('base_url')
    api = tradeapi.REST(api_key, api_secret, base_url)

    for ticker in tickers:
        logger.info(f"Downloading crypto data for {ticker} from {start_date} to {end_date}...")
        bars = _get_crypto_bars(api, ticker, start_date, end_date)
        rows = _bars_to_csv_rows(bars)
        if rows:
            df = pd.DataFrame(rows)
            # Sanitize ticker name for file system (replace / with -)
            safe_ticker = ticker.replace('/', '-')
            df.to_csv(f"./data/{safe_ticker}.csv", index=False)
            logger.info(f"Crypto data for {ticker} saved with {len(df)} rows.")
        else:
            logger.warning(f"No crypto data returned for {ticker}.")

def download_historical_data_with_crypto(tickers: List[str], config: dict, start_date: str = "2023-02-20", end_date: Optional[str] = None):
    """Download historical data for stocks (Polygon) and crypto (Alpaca)."""
    stock_tickers, crypto_tickers = split_tickers(tickers, config)
    if stock_tickers:
        download_historical_data(stock_tickers, config['polygon']['api_key'], start_date=start_date, end_date=end_date)

    if crypto_tickers:
        crypto_cfg = config.get('crypto', {})
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        lookback_days = crypto_cfg.get('lookback_days')
        crypto_start = start_date
        if lookback_days:
            try:
                crypto_start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=int(lookback_days))).strftime("%Y-%m-%d")
            except Exception as e:
                logger.warning(f"Invalid crypto lookback_days ({lookback_days}): {e}. Using start_date.")
        download_historical_crypto_data(crypto_tickers, config['alpaca'], start_date=crypto_start, end_date=end_date)

def download_historical_data(tickers: List[str], polygon_api_key: str, start_date: str = "2023-02-20", end_date: str = None):
    """Download historical market data from Polygon.io with rate limiting."""
    try:
        os.makedirs("./data", exist_ok=True)
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        headers = {"Authorization": f"Bearer {polygon_api_key}"}
        request_interval = 12  # 12 seconds between requests (5 calls/min)

        for ticker in tickers:
            logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}...")
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('results', [])
                if data:
                    df = pd.DataFrame([{
                        'time': datetime.fromtimestamp(bar['t']/1000).strftime('%Y-%m-%d'),
                        'open': bar['o'],
                        'high': bar['h'],
                        'low': bar['l'],
                        'close': bar['c'],
                        'volume': bar['v']
                    } for bar in data])
                    df.to_csv(f"./data/{ticker}.csv", index=False)
                    logger.info(f"Data for {ticker} downloaded and saved with {len(df)} rows.")
                else:
                    logger.warning(f"No data returned for {ticker}.")
            elif response.status_code == 429:
                logger.error(f"Rate limit exceeded for {ticker}. Waiting...")
                time.sleep(request_interval)
            elif response.status_code == 401:
                logger.error(f"Unauthorized access for {ticker}. Polygon API key: {polygon_api_key}")
            else:
                logger.error(f"Failed to download data for {ticker}. Status: {response.status_code}, Message: {response.text}")
            time.sleep(request_interval)
    except Exception as e:
        logger.error(f"Error downloading data for ticker {ticker}: {e}", exc_info=True)

def fetch_current_market_data(tickers: List[str], polygon_api_key: str, days: int = 200) -> pd.DataFrame:
    """Fetch recent market data from Polygon.io with enough history for features."""
    try:
        logger.info("Fetching current market data with Polygon.io...")
        headers = {"Authorization": f"Bearer {polygon_api_key}"}
        data_list = []
        request_interval = 12  # 12 seconds between requests (5 calls/min)

        end_time = datetime.now().strftime("%Y-%m-%d")
        start_time = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        for ticker in tickers:
            try:
                logger.debug(f"Fetching data for {ticker} from {start_time} to {end_time}...")
                url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_time}/{end_time}"
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    if results:
                        ticker_data = [{
                            'ticker': ticker,
                            'date': datetime.fromtimestamp(bar['t']/1000).strftime('%Y-%m-%d'),
                            'open': bar['o'],
                            'high': bar['h'],
                            'low': bar['l'],
                            'close': bar['c'],
                            'volume': bar['v']
                        } for bar in results]
                        logger.debug(f"Fetched {len(ticker_data)} rows for {ticker}")
                        data_list.extend(ticker_data)
                    else:
                        logger.warning(f"No recent data for {ticker}.")
                elif response.status_code == 401:
                    logger.error(f"Unauthorized access for {ticker}. Polygon API key: {polygon_api_key}")
                    return pd.DataFrame()
                elif response.status_code == 429:
                    logger.warning(f"Rate limit exceeded for {ticker}. Waiting...")
                    time.sleep(request_interval)
                else:
                    logger.warning(f"Failed to fetch data for {ticker}. Status: {response.status_code}, Message: {response.text}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error fetching data for {ticker}: {e}")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
            time.sleep(request_interval)

        if not data_list:
            logger.warning("No valid market data retrieved.")
            return pd.DataFrame()

        df = pd.DataFrame(data_list)
        logger.info(f"Market data fetched successfully. Total rows: {len(df)}")
        for ticker in tickers:
            rows = len(df[df['ticker'] == ticker])
            logger.info(f"Rows for {ticker}: {rows}")
        return df
    except Exception as e:
        logger.error(f"Error in fetch_current_market_data: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_current_crypto_market_data(tickers: List[str], api: tradeapi.REST, days: int = 200) -> pd.DataFrame:
    """Fetch recent crypto market data from Alpaca."""
    try:
        logger.info("Fetching current crypto market data from Alpaca...")
        data_list = []
        end_time = datetime.now().strftime("%Y-%m-%d")
        start_time = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        for ticker in tickers:
            bars = _get_crypto_bars(api, ticker, start_time, end_time)
            data_list.extend(_bars_to_rows(ticker, bars))
        if not data_list:
            logger.warning("No valid crypto market data retrieved.")
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        logger.info(f"Crypto market data fetched successfully. Total rows: {len(df)}")
        for ticker in tickers:
            rows = len(df[df['ticker'] == ticker])
            logger.info(f"Rows for {ticker}: {rows}")
        return df
    except Exception as e:
        logger.error(f"Error in fetch_current_crypto_market_data: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_current_market_data_with_crypto(tickers: List[str], polygon_api_key: str, api: tradeapi.REST, days: int = 200, config: Optional[dict] = None) -> pd.DataFrame:
    """Fetch recent market data for stocks (Polygon) and crypto (Alpaca)."""
    stock_tickers, crypto_tickers = split_tickers(tickers, config)
    frames = []
    if stock_tickers:
        frames.append(fetch_current_market_data(stock_tickers, polygon_api_key, days=days))
    if crypto_tickers:
        frames.append(fetch_current_crypto_market_data(crypto_tickers, api, days=days))
    if not frames:
        return pd.DataFrame()
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def fetch_latest_data(tickers: List[str], api: tradeapi.REST, config: Optional[dict] = None) -> pd.DataFrame:
    """Fetch latest daily bar data (partial if during market) from Alpaca API for real-time trading."""
    try:
        logger.info("Fetching latest daily bar data from Alpaca...")
        data_list = []
        for ticker in tickers:
            try:
                if is_crypto_ticker(ticker, config):
                    start_time = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
                    end_time = datetime.now().strftime("%Y-%m-%d")
                    bars = _get_crypto_bars(api, ticker, start_time, end_time)
                    bar = list(bars)[-1] if bars else None
                else:
                    bars = api.get_bars(ticker, tradeapi.TimeFrame(1, tradeapi.TimeFrameUnit.Day), limit=1)
                    bar = bars[0] if bars else None
                if bar:
                    data = {
                        'ticker': ticker,
                        'date': datetime.fromtimestamp(bar.t.timestamp()).strftime('%Y-%m-%d'),
                        'open': bar.o,
                        'high': bar.h,
                        'low': bar.l,
                        'close': bar.c,
                        'volume': bar.v
                    }
                    data_list.append(data)
                else:
                    logger.warning(f"No latest daily bar data for {ticker}.")
            except Exception as e:
                logger.error(f"Error fetching latest data for {ticker}: {e}")
        if not data_list:
            logger.warning("No valid latest data retrieved.")
            return pd.DataFrame()
        df = pd.DataFrame(data_list)
        logger.info(f"Latest data fetched successfully. Total rows: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error in fetch_latest_data: {e}", exc_info=True)
        return pd.DataFrame()

def load_historical_data(directory: str, config: dict) -> pd.DataFrame:
    """Load and combine historical data from CSV files or download if missing."""
    os.makedirs(directory, exist_ok=True)
    tickers = config['tickers']

    # Create safe ticker mapping (BTC/USD -> BTC-USD for filenames)
    safe_ticker_map = {ticker.replace('/', '-'): ticker for ticker in tickers}

    all_files = [f for f in os.listdir(directory) if f.endswith('.csv') and os.path.splitext(f)[0] in safe_ticker_map]
    logger.info(f"Files found in {directory} for tickers {tickers}: {all_files}")

    if not all_files:
        logger.warning("No CSV files found. Downloading historical data (Polygon for stocks, Alpaca for crypto)...")
        download_historical_data_with_crypto(tickers, config)
        all_files = [f for f in os.listdir(directory) if f.endswith('.csv') and os.path.splitext(f)[0] in safe_ticker_map]

    combined_data = []
    for file in all_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_csv(file_path)
            if 'time' in df.columns:
                df['date'] = df['time']  # Already str in '%Y-%m-%d'
                # Map safe filename back to original ticker (BTC-USD -> BTC/USD)
                safe_name = os.path.splitext(file)[0]
                df['ticker'] = safe_ticker_map.get(safe_name, safe_name)
                combined_data.append(df)
            else:
                logger.warning(f"File {file} does not contain a 'time' column. Skipping.")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}", exc_info=True)

    if combined_data:
        df = pd.concat(combined_data, ignore_index=True)
        logger.info(f"Loaded historical data with {len(df)} rows")
        return df
    else:
        logger.error("No valid historical data files found in the directory.")
        return pd.DataFrame()

async def fetch_news(config):
    if not config.get('newsapi_token'):
        logger.error("Missing NewsAPI token")
        return []
    url = "https://newsapi.org/v2/everything"
    params = {"q": " OR ".join(config['tickers']), "apiKey": config['newsapi_token'], "language": "en", "pageSize": 50}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                data = await resp.json()
                articles = data.get('articles', [])
                for article in articles:
                    title = article.get('title', '') or ''
                    description = article.get('description', '') or ''
                    text = (title + description).lower()
                    for t in config['tickers']:
                        if t.lower() in text:
                            article['ticker'] = t
                            break
                return [a for a in articles if 'ticker' in a]
    except Exception as e:
        logger.error(f"News fetch error: {e}")
        return []

def fetch_twitter_sync(config):
    logger.warning("Skipping Twitter sentiment: free API tier doesn’t support search.")
    return []

def analyze_sentiment(items, config, classification=True):
    """
    Analyze sentiment of news/social items using Claude (primary) or Grok (fallback).

    Features:
    - Batch processing: All headlines in single API call (much faster)
    - Claude primary: Fast, reliable, great reasoning
    - Grok fallback: Has X/Twitter data access
    - Recency weighting: Newer articles have more influence
    - Magnitude scoring: Strong/Moderate/Weak sentiment
    """
    sentiment_cfg = config.get('sentiment', {}) if isinstance(config, dict) else {}
    max_items = int(sentiment_cfg.get('max_items', 20))
    llm_timeout_seconds = float(sentiment_cfg.get('llm_timeout_seconds', 30))
    batch_mode = sentiment_cfg.get('batch_mode', True)
    primary_provider = sentiment_cfg.get('primary_provider', 'claude')
    fallback_provider = sentiment_cfg.get('fallback_provider', 'grok')

    # Prepare items with text
    prepared_items = []
    now = datetime.now()
    for it in items[:max_items]:
        text = it.get('text') or it.get('title') or it.get('description', '')
        if text:
            prepared_items.append({'item': it, 'text': text})

    if not prepared_items:
        logger.warning("No items to analyze for sentiment.")
        return []

    # Try primary provider first, then fallback
    providers = [primary_provider, fallback_provider]
    for provider in providers:
        try:
            if provider == 'claude':
                result = _analyze_sentiment_claude(prepared_items, config, llm_timeout_seconds, batch_mode, now)
            else:
                result = _analyze_sentiment_grok(prepared_items, config, llm_timeout_seconds, batch_mode, now)

            if result:
                logger.info(f"Sentiment analysis completed with {provider}: {len(result)} items")
                return result
        except Exception as e:
            logger.warning(f"{provider} sentiment failed: {e}, trying fallback...")

    logger.error("All sentiment providers failed.")
    return []


def _analyze_sentiment_claude(prepared_items, config, timeout, batch_mode, now):
    """Analyze sentiment using Claude API (Anthropic)."""
    import anthropic

    claude_cfg = config.get('claude', {})
    api_key = claude_cfg.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError("Claude API key not configured (config.yaml or ANTHROPIC_API_KEY env)")

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    model = claude_cfg.get('model', 'claude-3-haiku-20240307')

    if batch_mode:
        return _batch_sentiment_claude(client, model, prepared_items, now)
    else:
        return _sequential_sentiment_claude(client, model, prepared_items, now)


def _batch_sentiment_claude(client, model, prepared_items, now):
    """Process all headlines in a single Claude API call."""
    # Build batch prompt
    headlines = []
    for idx, pi in enumerate(prepared_items, 1):
        ticker = pi['item'].get('ticker', 'UNKNOWN')
        text = pi['text'][:200]  # Truncate long headlines
        headlines.append(f"{idx}. [{ticker}] {text}")

    batch_prompt = f"""Analyze the sentiment of each headline for stock trading.

Headlines:
{chr(10).join(headlines)}

For each headline, respond with EXACTLY this format (one per line):
NUMBER|TICKER|SENTIMENT|MAGNITUDE

Where:
- NUMBER: The headline number (1, 2, 3, etc.)
- TICKER: The stock ticker
- SENTIMENT: Positive, Negative, or Neutral
- MAGNITUDE: Strong, Moderate, or Weak

Example response:
1|AAPL|Positive|Strong
2|NVDA|Negative|Moderate
3|TSLA|Neutral|Weak

Respond with ONLY the formatted lines, nothing else."""

    resp = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": batch_prompt}]
    )

    # Parse batch response
    out = []
    response_text = resp.content[0].text.strip()

    for line in response_text.split('\n'):
        line = line.strip()
        if not line or '|' not in line:
            continue

        parts = line.split('|')
        if len(parts) >= 4:
            try:
                idx = int(parts[0]) - 1
                if 0 <= idx < len(prepared_items):
                    ticker = parts[1].strip()
                    label = parts[2].strip().capitalize()
                    magnitude_str = parts[3].strip().lower()

                    # Normalize label
                    if 'positive' in label.lower():
                        label = 'Positive'
                    elif 'negative' in label.lower():
                        label = 'Negative'
                    else:
                        label = 'Neutral'

                    # Parse magnitude
                    if 'strong' in magnitude_str:
                        magnitude = 1.5
                    elif 'weak' in magnitude_str:
                        magnitude = 0.5
                    else:
                        magnitude = 1.0

                    # Calculate recency weight
                    recency_weight = _calculate_recency_weight(prepared_items[idx]['item'], now)

                    out.append({
                        'ticker': prepared_items[idx]['item'].get('ticker', ticker),
                        'label': label,
                        'magnitude': magnitude,
                        'recency_weight': recency_weight,
                        'detail': f"{label}|{magnitude_str.capitalize()}"
                    })
            except (ValueError, IndexError) as e:
                logger.debug(f"Skipping malformed line: {line}")

    logger.info(f"Claude batch processed {len(out)}/{len(prepared_items)} items")
    return out


def _sequential_sentiment_claude(client, model, prepared_items, now):
    """Process headlines one by one (fallback if batch parsing fails)."""
    out = []
    sys_prompt = (
        "Analyze the sentiment and respond in format: LABEL|MAGNITUDE\n"
        "LABEL: Positive, Neutral, or Negative\n"
        "MAGNITUDE: Strong, Moderate, or Weak\n"
        "Example: Positive|Strong\n"
        "Only respond with the format, nothing else."
    )

    for idx, pi in enumerate(prepared_items, 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=50,
                system=sys_prompt,
                messages=[{"role": "user", "content": pi['text']}]
            )
            msg = resp.content[0].text.strip().lower()
            label, magnitude = _parse_sentiment_response(msg)
            recency_weight = _calculate_recency_weight(pi['item'], now)

            out.append({
                'ticker': pi['item'].get('ticker', 'UNKNOWN'),
                'label': label,
                'magnitude': magnitude,
                'recency_weight': recency_weight,
                'detail': msg
            })
        except Exception as e:
            logger.error(f"Claude error for: {pi['text'][:50]}...: {e}")

        if idx % 10 == 0:
            logger.info(f"Sentiment progress: {idx}/{len(prepared_items)}")

    return out


def _analyze_sentiment_grok(prepared_items, config, timeout, batch_mode, now):
    """Analyze sentiment using Grok API (xAI) - fallback provider."""
    if not config.get('grok_api_key'):
        raise ValueError("Grok API key not configured")

    client = OpenAI(
        api_key=config['grok_api_key'],
        base_url='https://api.x.ai/v1',
        timeout=timeout,
    )

    if batch_mode:
        return _batch_sentiment_grok(client, prepared_items, now)
    else:
        return _sequential_sentiment_grok(client, prepared_items, now)


def _batch_sentiment_grok(client, prepared_items, now):
    """Process all headlines in a single Grok API call."""
    headlines = []
    for idx, pi in enumerate(prepared_items, 1):
        ticker = pi['item'].get('ticker', 'UNKNOWN')
        text = pi['text'][:200]
        headlines.append(f"{idx}. [{ticker}] {text}")

    batch_prompt = f"""Analyze sentiment for trading. Headlines:

{chr(10).join(headlines)}

Respond with EXACTLY this format (one per line):
NUMBER|TICKER|SENTIMENT|MAGNITUDE

SENTIMENT: Positive, Negative, or Neutral
MAGNITUDE: Strong, Moderate, or Weak

Example: 1|AAPL|Positive|Strong

ONLY formatted lines, nothing else."""

    try:
        resp = client.chat.completions.create(
            model='grok-4',
            messages=[{'role': 'user', 'content': batch_prompt}],
            temperature=0,
            max_tokens=1000
        )

        out = []
        response_text = resp.choices[0].message.content.strip()

        for line in response_text.split('\n'):
            line = line.strip()
            if not line or '|' not in line:
                continue

            parts = line.split('|')
            if len(parts) >= 4:
                try:
                    idx = int(parts[0]) - 1
                    if 0 <= idx < len(prepared_items):
                        label, magnitude = _parse_sentiment_response(f"{parts[2]}|{parts[3]}")
                        recency_weight = _calculate_recency_weight(prepared_items[idx]['item'], now)

                        out.append({
                            'ticker': prepared_items[idx]['item'].get('ticker', parts[1].strip()),
                            'label': label,
                            'magnitude': magnitude,
                            'recency_weight': recency_weight,
                            'detail': f"{label}|{parts[3].strip()}"
                        })
                except (ValueError, IndexError):
                    pass

        logger.info(f"Grok batch processed {len(out)}/{len(prepared_items)} items")
        return out

    except Exception as e:
        logger.error(f"Grok batch failed: {e}")
        return _sequential_sentiment_grok(client, prepared_items, now)


def _sequential_sentiment_grok(client, prepared_items, now):
    """Process headlines one by one with Grok (original method)."""
    out = []
    sys_prompt = (
        "Analyze the sentiment and respond in format: LABEL|MAGNITUDE\n"
        "LABEL: Positive, Neutral, or Negative\n"
        "MAGNITUDE: Strong, Moderate, or Weak\n"
        "Example: Positive|Strong\n"
        "Only respond with the format, nothing else."
    )

    for idx, pi in enumerate(prepared_items, 1):
        try:
            resp = client.chat.completions.create(
                model='grok-4',
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': pi['text']}
                ],
                temperature=0,
                max_tokens=50
            )
            msg = resp.choices[0].message.content.strip().lower()
            label, magnitude = _parse_sentiment_response(msg)
            recency_weight = _calculate_recency_weight(pi['item'], now)

            out.append({
                'ticker': pi['item'].get('ticker', 'UNKNOWN'),
                'label': label,
                'magnitude': magnitude,
                'recency_weight': recency_weight,
                'detail': msg
            })
        except Exception as e:
            logger.error(f"Grok error for: {pi['text'][:50]}...: {e}")

        if idx % 10 == 0:
            logger.info(f"Sentiment progress: {idx}/{len(prepared_items)}")

    return out


def _parse_sentiment_response(msg):
    """Parse sentiment label and magnitude from LLM response."""
    msg_lower = msg.lower()

    # Parse label
    if 'positive' in msg_lower:
        label = 'Positive'
    elif 'negative' in msg_lower:
        label = 'Negative'
    else:
        label = 'Neutral'

    # Parse magnitude
    if 'strong' in msg_lower:
        magnitude = 1.5
    elif 'weak' in msg_lower:
        magnitude = 0.5
    else:
        magnitude = 1.0

    return label, magnitude


def _calculate_recency_weight(item, now):
    """Calculate recency weight for a news item (newer = higher weight)."""
    published_at = item.get('publishedAt', '')
    try:
        if published_at:
            pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            hours_ago = (now - pub_time.replace(tzinfo=None)).total_seconds() / 3600
            # Exponential decay: weight = exp(-hours/24)
            return max(0.1, min(1.0, np.exp(-hours_ago / 24)))
    except:
        pass
    return 0.5  # Unknown time, use medium weight

# =============================================================================
# Utility Functions
# =============================================================================

def send_notification(title, message):
    script = f'display notification "{message}" with title "{title}"'
    os.system(f"osascript -e '{script}'")

def initialize_trade_log() -> str:
    """Initialize a CSV file to log trades."""
    trades_dir = os.path.join(LOG_DIR, "trade_logs")
    os.makedirs(trades_dir, exist_ok=True)
    log_filename = os.path.join(trades_dir, f"trade_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    try:
        with open(log_filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'ticker', 'type', 'price', 'quantity', 'status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        logger.info(f"Trade log initialized: {log_filename}")
    except Exception as e:
        logger.error(f"Failed to initialize trade log: {e}", exc_info=True)
    return log_filename

def log_trade(trade_data: dict, log_filename: str):
    """Log individual trades into the CSV file."""
    try:
        with open(log_filename, mode='a', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow([
                datetime.now(), trade_data['ticker'], trade_data['type'],
                trade_data['price'], trade_data['quantity'], trade_data['status']
            ])
        logger.info(f"Trade logged: {trade_data}")
    except Exception as e:
        logger.error(f"Failed to log trade: {e}", exc_info=True)

def fetch_current_positions(api: tradeapi.REST) -> list:
    """Fetch current positions from Alpaca API with timeout handling."""
    try:
        positions = api.list_positions()
        current_positions = []
        for position in positions:
            ticker = position.symbol
            qty = float(position.qty)
            avg_price = float(position.avg_entry_price)
            try:
                current_price = float(api.get_latest_trade(ticker).price)
            except Exception as e:
                logger.debug(f"Error fetching current price for {ticker}: {e}")
                current_price = avg_price  # Fallback to avg_price if API fails
            current_positions.append({
                'ticker': ticker,
                'quantity': qty,
                'average_price': avg_price,
                'current_price': current_price
            })
            logger.debug(f"Current Position: {ticker} - {qty} shares at an average price of {avg_price}, current price {current_price}")
        return current_positions
    except Exception as e:
        # Reduce log level to debug for timeout errors (common during off-hours)
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            logger.debug(f"Alpaca API timeout fetching positions (likely market closed): {e}")
        else:
            logger.error(f"Error fetching current positions: {e}", exc_info=True)
        return []

def is_market_open():
    """Check if the market is open (9:30 AM - 4:00 PM ET)."""
    now = datetime.now(tz=pytz.timezone('US/Eastern'))
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    return now.weekday() < 5 and market_open <= now.time() <= market_close

def is_crypto_ticker(ticker: str, config: Optional[dict] = None) -> bool:
    """Return True when ticker is configured/recognized as crypto."""
    if not ticker:
        return False
    if config:
        crypto_cfg = config.get('crypto', {}) if isinstance(config, dict) else {}
        if crypto_cfg.get('enabled', False) is not True:
            return False
        crypto_tickers = set(map(str.upper, crypto_cfg.get('tickers', [])))
        if ticker.upper() in crypto_tickers:
            return True
    ticker_upper = ticker.upper()
    crypto_suffixes = ("USD", "USDT", "USDC")
    return any(ticker_upper.endswith(suffix) for suffix in crypto_suffixes)

def split_tickers(tickers: List[str], config: Optional[dict] = None) -> tuple[List[str], List[str]]:
    """Split tickers into stock and crypto lists."""
    stock_tickers = []
    crypto_tickers = []
    for ticker in tickers or []:
        if is_crypto_ticker(ticker, config):
            crypto_tickers.append(ticker)
        else:
            stock_tickers.append(ticker)
    return stock_tickers, crypto_tickers

def get_asset_config(ticker: str, config: dict) -> dict:
    """Return asset-specific config (crypto overrides when applicable)."""
    if is_crypto_ticker(ticker, config):
        return config.get('crypto', {}) if isinstance(config, dict) else {}
    return config

def maybe_add_crypto_tickers(config: dict) -> dict:
    """Merge crypto tickers into the main ticker list when enabled."""
    if not isinstance(config, dict):
        return config
    crypto_cfg = config.get('crypto', {})
    if crypto_cfg.get('enabled', False) is not True:
        return config
    crypto_tickers = crypto_cfg.get('tickers', [])
    if not crypto_tickers:
        return config
    tickers = list(config.get('tickers', []))
    for ticker in crypto_tickers:
        if ticker not in tickers:
            tickers.append(ticker)
    config['tickers'] = tickers
    return config

# =============================================================================
# Feature Engineering (Combined from both scripts)
# =============================================================================

# Cache for FinViz fundamentals to avoid repeated API calls
_finviz_cache = {}
_finviz_cache_time = None

def fetch_finviz_fundamentals(tickers: List[str], cache_minutes: int = 60) -> dict:
    """
    Fetch fundamental data from FinViz for all tickers.
    Uses caching to avoid excessive API calls.

    Returns dict mapping ticker -> fundamental features dict
    """
    global _finviz_cache, _finviz_cache_time

    # Check cache validity
    if _finviz_cache_time and (datetime.now() - _finviz_cache_time).total_seconds() < cache_minutes * 60:
        # Return cached data if still valid
        missing = [t for t in tickers if t not in _finviz_cache]
        if not missing:
            logger.debug("Using cached FinViz fundamentals")
            return {t: _finviz_cache[t] for t in tickers if t in _finviz_cache}

    try:
        from finviz_enrichment import get_fundamentals_for_ml, FINVIZ_AVAILABLE
        if not FINVIZ_AVAILABLE:
            logger.debug("FinViz not available, skipping fundamental features")
            return {}

        logger.info(f"Fetching FinViz fundamentals for {len(tickers)} tickers...")
        fundamentals = get_fundamentals_for_ml(tickers)

        # Update cache
        _finviz_cache.update(fundamentals)
        _finviz_cache_time = datetime.now()

        logger.info(f"FinViz fundamentals fetched for {len(fundamentals)} tickers")
        return fundamentals

    except Exception as e:
        logger.warning(f"Failed to fetch FinViz fundamentals: {e}")
        return {}


def add_fundamental_features(df: pd.DataFrame, fundamentals: dict) -> pd.DataFrame:
    """
    Add FinViz fundamental features to the DataFrame.

    Each ticker gets its fundamental ratios added as static columns.
    """
    if df.empty or not fundamentals:
        return df

    # Define which fundamental features to add to ML
    fundamental_cols = [
        'PE_Ratio', 'Forward_PE', 'PEG_Ratio', 'Debt_Equity',
        'ROE', 'Profit_Margin', 'Short_Float', 'Beta',
        'Analyst_Recom', 'SMA20_Dist', 'SMA50_Dist'
    ]

    # Initialize columns with NaN
    for col in fundamental_cols:
        df[col] = np.nan

    # Fill in fundamentals for each ticker
    for ticker in df['ticker'].unique():
        if ticker in fundamentals:
            fund = fundamentals[ticker]
            mask = df['ticker'] == ticker
            for col in fundamental_cols:
                if col in fund and fund[col] is not None:
                    df.loc[mask, col] = fund[col]

    # Fill NaN with reasonable defaults
    defaults = {
        'PE_Ratio': 20,        # Market average
        'Forward_PE': 18,
        'PEG_Ratio': 1.5,
        'Debt_Equity': 0.5,
        'ROE': 15,
        'Profit_Margin': 10,
        'Short_Float': 3,
        'Beta': 1.0,
        'Analyst_Recom': 2.5,  # Hold
        'SMA20_Dist': 0,
        'SMA50_Dist': 0,
    }

    for col, default in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    return df


def engineer_features(df: pd.DataFrame, include_fundamentals: bool = True) -> pd.DataFrame:
    """Perform feature engineering on the dataset, combining indicators from both scripts."""
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping feature engineering.")
        return pd.DataFrame()

    df.sort_values(['ticker', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df_list = []
    for ticker in df['ticker'].unique():
        df_ticker = df[df['ticker'] == ticker].copy()

        logger.debug(f"Processing {ticker} with {len(df_ticker)} rows")
        if len(df_ticker) < 5:  # Reduced from 50 for testing
            logger.warning(f"Not enough data for {ticker} ({len(df_ticker)} rows). Skipping.")
            continue

        # From Trader_main_Grok3.py
        df_ticker['MA10'] = df_ticker['close'].rolling(window=10, min_periods=1).mean()
        df_ticker['MA50'] = df_ticker['close'].rolling(window=50, min_periods=1).mean()
        df_ticker['RSI'] = ta.momentum.RSIIndicator(df_ticker['close'], window=14).rsi()

        macd_indicator = ta.trend.MACD(df_ticker['close'])
        df_ticker['MACD'] = macd_indicator.macd()
        df_ticker['MACD_Signal'] = macd_indicator.macd_signal()
        df_ticker['MACD_Diff'] = macd_indicator.macd_diff()

        bollinger = ta.volatility.BollingerBands(df_ticker['close'])
        df_ticker['Bollinger_Upper'] = bollinger.bollinger_hband()
        df_ticker['Bollinger_Lower'] = bollinger.bollinger_lband()

        try:
            atr_indicator = ta.volatility.AverageTrueRange(df_ticker['high'], df_ticker['low'], df_ticker['close'], window=5)  # Reduced window
            df_ticker['ATR'] = atr_indicator.average_true_range()
        except IndexError as e:
            logger.warning(f"Insufficient data for ATR calculation on {ticker}: {e}. Defaulting to 1.0.")
            df_ticker['ATR'] = 1.0

        stoch_rsi_indicator = ta.momentum.StochRSIIndicator(df_ticker['close'])
        df_ticker['Stochastic_RSI'] = stoch_rsi_indicator.stochrsi()

        df_ticker['Lag1_Close'] = df_ticker['close'].shift(1)
        df_ticker['Lag2_Close'] = df_ticker['close'].shift(2)

        # Volatility Regime Detection (Grok 4.20 enhancement)
        # Calculate rolling volatility percentile for adaptive position sizing
        df_ticker['Returns'] = df_ticker['close'].pct_change()
        df_ticker['Volatility_20d'] = df_ticker['Returns'].rolling(window=20, min_periods=5).std()

        # Bollinger Band Width as volatility proxy
        bb_width = (df_ticker['Bollinger_Upper'] - df_ticker['Bollinger_Lower']) / df_ticker['close']
        df_ticker['BB_Width'] = bb_width

        # Volatility regime: 0=low, 1=medium, 2=high (based on 60-day percentile rank)
        vol_percentile = df_ticker['Volatility_20d'].rolling(window=60, min_periods=10).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
        )
        df_ticker['Vol_Percentile'] = vol_percentile.fillna(0.5)
        df_ticker['Vol_Regime'] = pd.cut(
            df_ticker['Vol_Percentile'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=[0, 1, 2],
            include_lowest=True
        ).astype(float).fillna(1)

        # Trend strength using ADX-like calculation (for adaptive entry thresholds)
        high_low = df_ticker['high'] - df_ticker['low']
        high_close = abs(df_ticker['high'] - df_ticker['close'].shift(1))
        low_close = abs(df_ticker['low'] - df_ticker['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14, min_periods=1).mean()

        plus_dm = df_ticker['high'].diff()
        minus_dm = -df_ticker['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        df_ticker['ADX'] = dx.rolling(window=14, min_periods=1).mean().fillna(25)

        # Price momentum for sentiment correlation
        df_ticker['Price_Momentum_5d'] = df_ticker['close'].pct_change(5)

        # From trade_bot.py (additional indicators if TA-Lib available)
        if talib:
            df_ticker['Momentum'] = talib.MOM(df_ticker['close'], timeperiod=10)
            df_ticker['SMA_20'] = talib.SMA(df_ticker['close'], timeperiod=20)

        if len(df_ticker['volume']) > 1:
            df_ticker['Volume_Change'] = df_ticker['volume'] / df_ticker['volume'].shift(1)

        df_ticker['buy_signal'] = (
            (df_ticker['RSI'] < 30) &
            (df_ticker['MACD'] > df_ticker['MACD_Signal']) &
            (df_ticker['Stochastic_RSI'] < 0.2)
        ).astype(int)

        df_ticker['sell_signal'] = (
            (df_ticker['RSI'] > 70) &
            (df_ticker['MACD'] < df_ticker['MACD_Signal']) &
            (df_ticker['Stochastic_RSI'] > 0.8)
        ).astype(int)

        df_ticker['Target'] = (df_ticker['close'].shift(-1) > df_ticker['close']).astype(int)

        df_ticker.dropna(inplace=True)
        logger.debug(f"After NaN drop for {ticker}: {len(df_ticker)} rows")  # Log post-NaN count
        if len(df_ticker) < 5:
            logger.warning(f"Insufficient rows after NaN drop for {ticker} ({len(df_ticker)}). Skipping.")
            continue
        logger.debug(f"After feature engineering, {ticker} has {len(df_ticker)} rows")
        df_list.append(df_ticker)
        
    if df_list:
        df = pd.concat(df_list)
        logger.info(f"Engineered DataFrame shape: {df.shape}")  # Log shape
        logger.info(f"NaNs in DataFrame:\n{df.isna().sum()}")  # Log NaNs
        df.reset_index(drop=True, inplace=True)
    
        # Merge VIX if present (assume '^VIX' data fetched)
        vix_data = df[df['ticker'] == '^VIX'][['date', 'close']].rename(columns={'close': 'VIX'})
    
        if not vix_data.empty:
            df = df.merge(vix_data, on='date', how='left')
            df = df[df['ticker'] != '^VIX']  # Remove VIX row
        else:
            df['VIX'] = 0  # Fallback if no VIX data

        # Add FinViz fundamental features if enabled
        if include_fundamentals:
            try:
                tickers = [t for t in df['ticker'].unique() if not t.startswith('^') and '/' not in t]
                if tickers:
                    fundamentals = fetch_finviz_fundamentals(tickers)
                    if fundamentals:
                        df = add_fundamental_features(df, fundamentals)
                        logger.info(f"Added fundamental features for {len(fundamentals)} tickers")
            except Exception as e:
                logger.warning(f"Failed to add fundamental features: {e}")

        logger.info("Feature engineering completed successfully.")
        return df
    else:
        logger.warning("No valid tickers after feature engineering. Returning empty DataFrame.")
        return pd.DataFrame()


def add_sentiment_features(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Add sentiment scores as features to the DataFrame.

    Grok 4.20 Enhancement:
    - Recency-weighted sentiment scores
    - Magnitude-adjusted scores (strong sentiment counts more)
    - Sentiment momentum tracking (improving vs deteriorating)
    - Sentiment-price divergence detection
    """
    if df is None or df.empty:
        logger.warning("Input DataFrame is None or empty. Returning as is.")
        return df

    sentiment_cfg = config.get('sentiment', {}) if isinstance(config, dict) else {}
    if sentiment_cfg.get('enabled', True) is False:
        df['Sentiment_Score'] = 0.0
        df['Sentiment_Magnitude'] = 0.0
        df['Sentiment_Momentum'] = 0.0
        logger.info("Sentiment disabled via config; using Sentiment_Score=0.0")
        return df

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    news = loop.run_until_complete(fetch_news(config))
    loop.close()
    twitter = fetch_twitter_sync(config)
    sentiments = analyze_sentiment(news + twitter, config)

    sentiment_scores = {}
    sentiment_magnitudes = {}

    for ticker in config['tickers']:
        ticker_sentiments = [s for s in sentiments if s['ticker'] == ticker]
        if ticker_sentiments:
            # Grok 4.20: Weighted sentiment calculation
            total_weight = 0
            weighted_score = 0

            for s in ticker_sentiments:
                # Base score: Positive=1, Neutral=0, Negative=-1
                if s['label'] == 'Positive':
                    base_score = 1
                elif s['label'] == 'Negative':
                    base_score = -1
                else:
                    base_score = 0

                # Apply magnitude and recency weights
                magnitude = s.get('magnitude', 1.0)
                recency = s.get('recency_weight', 0.5)

                # Combined weight: magnitude * recency
                weight = magnitude * recency
                total_weight += weight
                weighted_score += base_score * weight

            # Normalized weighted sentiment score [-1, 1]
            sentiment_scores[ticker] = weighted_score / total_weight if total_weight > 0 else 0

            # Average magnitude (how strong is the sentiment overall)
            sentiment_magnitudes[ticker] = sum(s.get('magnitude', 1.0) for s in ticker_sentiments) / len(ticker_sentiments)
        else:
            sentiment_scores[ticker] = 0
            sentiment_magnitudes[ticker] = 0

    df['Sentiment_Score'] = df['ticker'].map(sentiment_scores)
    df['Sentiment_Magnitude'] = df['ticker'].map(sentiment_magnitudes)

    # Grok 4.20: Sentiment-Price Divergence Detection
    # If price momentum and sentiment are moving opposite directions, it's a signal
    if 'Price_Momentum_5d' in df.columns:
        # Divergence: positive price momentum + negative sentiment (or vice versa)
        df['Sentiment_Price_Divergence'] = df.apply(
            lambda row: (
                1 if (row.get('Price_Momentum_5d', 0) > 0.02 and row.get('Sentiment_Score', 0) < -0.2) else
                -1 if (row.get('Price_Momentum_5d', 0) < -0.02 and row.get('Sentiment_Score', 0) > 0.2) else
                0
            ), axis=1
        )
    else:
        df['Sentiment_Price_Divergence'] = 0

    logger.info(f"DataFrame shape after sentiment: {df.shape}")
    logger.info(f"Sentiment scores: {sentiment_scores}")
    return df

# =============================================================================
# Model Training and Evaluation (from Trader_main_Grok3.py)
# =============================================================================

def prepare_train_test_data(df: pd.DataFrame, selected_features: list):
    """Split the data into training and testing sets using time series split."""
    try:
        if 'date' in df.columns:
            df = df.sort_values('date')
        X = df[selected_features].reset_index(drop=True)
        y = df['Target'].reset_index(drop=True)

        total_samples = len(df)
        train_size = int(0.8 * total_samples)
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]

        logger.info(f"Length of X_train: {len(X_train)}, Length of y_train: {len(y_train)}")
        logger.info(f"Length of X_test: {len(X_test)}, Length of y_test: {len(y_test)}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in prepare_train_test_data: {e}", exc_info=True)
        raise

def tune_and_train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Perform hyperparameter tuning and train the model."""
    try:
        param_grid_xgb = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'scale_pos_weight': [1, 3, 5, 7],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }

        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = RandomizedSearchCV(
            XGBClassifier(eval_metric='logloss'),
            param_distributions=param_grid_xgb,
            n_iter=100,
            cv=tscv,
            scoring='precision',
            n_jobs=-1,
            random_state=42
        )
        logger.info("Starting hyperparameter tuning with RandomizedSearchCV...")
        
        if len(X_train) > 0:
            grid_search.fit(X_train, y_train)
            logger.info("Hyperparameter tuning completed.")
            final_model = grid_search.best_estimator_
            joblib.dump(final_model, os.path.join(ARTIFACT_DIR, 'final_model.pkl'))
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info("Final model saved as 'artifacts/final_model.pkl'.")
            return final_model
        else:
            logger.warning("No training data available. Model training skipped.")
            return None
    except Exception as e:
        logger.error(f"Error in tune_and_train_model: {e}", exc_info=True)
        raise

def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate the trained model and save confusion matrix."""
    try:
        if X_test.isnull().values.any():
            logger.error("NaN values found in X_test. Dropping NaNs.")
            X_test = X_test.dropna()
            y_test = y_test.loc[X_test.index]
        if y_test.isnull().values.any():
            logger.error("NaN values found in y_test. Dropping NaNs.")
            y_test = y_test.dropna()
            X_test = X_test.loc[y_test.index]

        y_pred = model.predict(X_test)
        logger.info(f"Length of y_test: {len(y_test)}, Length of y_pred: {len(y_pred)}")
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(ARTIFACT_DIR, 'confusion_matrix.png'))
        plt.close()
        logger.info("Confusion matrix saved as 'artifacts/confusion_matrix.png'.")
        logger.info(f"Classification Report:\n{cr}")
        print(f"Classification Report:\n{cr}")
        try:
            with open(os.path.join(ARTIFACT_DIR, 'classification_report.txt'), 'w') as f:
                f.write(cr)
            logger.info("Classification report saved as 'artifacts/classification_report.txt'.")
        except Exception as e:
            logger.warning(f"Failed to save classification_report.txt: {e}")
    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}", exc_info=True)
        raise

# =============================================================================
# Q-Learning Integration (from trade_bot.py)
# =============================================================================

def load_q_table(path: str = None):
    try:
        path = path or os.path.join(ARTIFACT_DIR, 'q_table.csv')
        q_table = pd.read_csv(path, index_col='state')
        if q_table.empty or 'state' not in q_table.index.names:
            raise ValueError("Invalid Q-table format")
        # Ensure all columns exist
        expected_columns = ['hold', 'buy', 'sell']
        for col in expected_columns:
            if col not in q_table.columns:
                q_table[col] = 0.0
        return q_table
    except Exception as e:
        logger.warning(f"Q-table not found or invalid. Creating new table. {e}")
        states = ['-'.join(map(str, combo)) for combo in product(range(3), range(3), range(3), range(3), range(2))]
        q_table = pd.DataFrame(0.0, index=states, columns=['hold', 'buy', 'sell'])
        q_table.index.name = 'state'
        q_table.to_csv(path)
        return q_table

def load_last_decisions(path: str = None):
    try:
        path = path or os.path.join(ARTIFACT_DIR, 'last_decisions.csv')
        df = pd.read_csv(path)
        return df.set_index('ticker').to_dict('index')
    except Exception as e:
        logger.warning(f"Last decisions not found or invalid. {e}")
        return {}

def save_last_decisions(decisions, path: str = None):
    path = path or os.path.join(ARTIFACT_DIR, 'last_decisions.csv')
    if decisions:
        df = pd.DataFrame.from_dict(decisions, orient='index')
        df.to_csv(path, index_label='ticker')

def get_state(sentiment_score, indicators, holding_state):
    if sentiment_score < -0.1:
        sentiment_state = 0
    elif sentiment_score > 0.1:
        sentiment_state = 2
    else:
        sentiment_state = 1
    rsi = indicators.get('RSI', 50)
    if rsi < 30:
        rsi_state = 0
    elif rsi > 70:
        rsi_state = 2
    else:
        rsi_state = 1
    macd_hist = indicators.get('MACD_Diff', 0)
    if macd_hist < -0.1:
        macd_state = 0
    elif macd_hist > 0.1:
        macd_state = 2
    else:
        macd_state = 1
    volume_change = indicators.get('Volume_Change', 1)
    if volume_change < 0.8:
        volume_state = 0
    elif volume_change > 1.5:
        volume_state = 2
    else:
        volume_state = 1
    return '-'.join(map(str, [sentiment_state, rsi_state, macd_state, volume_state, holding_state]))

def select_action(q_table, state, holding_state, epsilon=0.05):
    if state not in q_table.index:
        return 'hold'
    valid_actions = ['hold']
    if holding_state == 0:
        valid_actions.append('buy')
    else:
        valid_actions.append('sell')
    if np.random.random() < epsilon:
        return np.random.choice(valid_actions)
    else:
        q_values = q_table.loc[state][valid_actions]
        return q_values.idxmax()

# =============================================================================
# Signal Generation and Trading Logic (Hybrid)
# =============================================================================

def generate_signals(model: XGBClassifier, data: pd.DataFrame, selected_features: List[str], config, q_table, threshold: float = 0.5, positions_snapshot: dict = None) -> pd.DataFrame:
    """Generate buy/sell signals using XGBoost predictions and refine with Q-learning.

    Grok 4.20 Enhancements:
    - Adaptive confidence thresholds based on volatility regime
    - Continuous VIX scaling instead of binary cutoff
    - Trend strength confirmation via ADX
    """
    if data.empty:
        logger.warning("generate_signals received empty DataFrame. Exiting early.")
        return pd.DataFrame()

    missing_features = [f for f in selected_features if f not in data.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}. Skipping signal generation.")
        return pd.DataFrame()

    try:
        probabilities = model.predict_proba(data[selected_features])[:, 1]
        data['Probability'] = probabilities

        # Adaptive threshold based on volatility regime (Grok 4.20 enhancement)
        # High vol (regime 2): require higher confidence (0.60-0.65)
        # Low vol (regime 0): accept lower confidence (0.45)
        # Medium vol (regime 1): standard threshold (0.50)
        vol_regime = data.get('Vol_Regime', pd.Series([1] * len(data)))
        adaptive_threshold = vol_regime.map({0: 0.45, 1: 0.50, 2: 0.60}).fillna(0.50)

        # ADX trend confirmation: require stronger signals in weak trends
        adx = data.get('ADX', pd.Series([25] * len(data)))
        # Weak trend (ADX < 20): increase threshold by 0.05
        # Strong trend (ADX > 30): decrease threshold by 0.03
        adx_adjustment = adx.apply(lambda x: 0.05 if x < 20 else (-0.03 if x > 30 else 0))
        adaptive_threshold = adaptive_threshold + adx_adjustment
        adaptive_threshold = adaptive_threshold.clip(0.40, 0.70)  # Safety bounds

        data['Adaptive_Threshold'] = adaptive_threshold
        data['Prediction'] = (probabilities >= adaptive_threshold).astype(int)
        data['Signal'] = data.groupby('ticker')['Prediction'].diff().fillna(0)
        data['Signal'] = data['Signal'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        # Continuous VIX scaling instead of binary cutoff (Grok 4.20 enhancement)
        # Scale signal confidence down as VIX increases above 20
        vix_threshold = config.get('vix_threshold', 25)
        for idx, row in data.iterrows():
            vix = row.get('VIX', 0)
            if vix > vix_threshold + 10:
                # VIX > 35: block all trades
                data.loc[idx, 'Signal'] = 0
            elif vix > vix_threshold:
                # VIX 25-35: only allow high-confidence trades (prob > 0.65)
                if probabilities[idx] < 0.65:
                    data.loc[idx, 'Signal'] = 0
            elif vix > 20:
                # VIX 20-25: caution zone, require prob > 0.55
                if probabilities[idx] < 0.55:
                    data.loc[idx, 'Signal'] = 0

        # Refine with Q-learning
        last_decisions = load_last_decisions()
        decisions_dict = {}
        positions_snapshot = positions_snapshot or {}
        for ticker in config['tickers']:
            ticker_data = data[data['ticker'] == ticker]
            if ticker_data.empty:
                continue
            sentiment_score = ticker_data['Sentiment_Score'].iloc[-1] if 'Sentiment_Score' in ticker_data.columns else 0
            indicators = ticker_data.iloc[-1].to_dict()
            position_info = positions_snapshot.get(ticker, {})
            holding_state = 1 if position_info.get('quantity', 0) > 0 else 0
            state = get_state(sentiment_score, indicators, holding_state)
            current_price = ticker_data['close'].iloc[-1]
            last = last_decisions.get(ticker)
            if last:
                last_state = last['state']
                last_action = last['action']
                last_price = last['price']
                last_state_split = last_state.split('-')
                last_holding_state = int(last_state_split[-1])
                delta = (current_price - last_price) / last_price
                if last_action == 'sell':
                    reward = 0.0
                elif last_action == 'buy':
                    reward = delta
                elif last_action == 'hold':
                    reward = delta if last_holding_state == 1 else 0.0
                else:
                    reward = 0.0
                alpha = 0.1
                gamma = 0.9
                if last_state in q_table.index and last_action in q_table.columns:
                    old_q = q_table.loc[last_state, last_action]
                    max_q_next = q_table.loc[state].max()
                    new_q = old_q + alpha * (reward + gamma * max_q_next - old_q)
                    q_table.loc[last_state, last_action] = new_q
                else:
                    logger.warning(f"State {last_state} or action {last_action} not found in Q-table. Skipping update.")
            action = select_action(q_table, state, holding_state, epsilon=0.05)
            # Override XGBoost signal with Q-learning if conflict
            if action == 'buy' and data.loc[data['ticker'] == ticker, 'Signal'].iloc[-1] != 1:
                data.loc[data['ticker'] == ticker, 'Signal'] = 1
            elif action == 'sell' and data.loc[data['ticker'] == ticker, 'Signal'].iloc[-1] != -1:
                data.loc[data['ticker'] == ticker, 'Signal'] = -1
            decisions_dict[ticker] = {'state': state, 'action': action, 'price': current_price}
            logger.info(f"{ticker}: State={state}, Action={action}, Sentiment={sentiment_score:.2f}")
        q_table.to_csv(os.path.join(ARTIFACT_DIR, 'q_table.csv'))
        save_last_decisions(decisions_dict)
        logger.info("Hybrid signals generated successfully.")
        return data
    except Exception as e:
        logger.error(f"Error in generate_signals: {e}", exc_info=True)
        return pd.DataFrame()

def simulate_trades(data: pd.DataFrame, initial_capital: float, config, max_position: int = 100,
                    stop_loss_pct: float = 0.05,
                    take_profit_pct: float = 0.10,
                    buying_power_pct: float = 50,
                    transaction_cost_pct: float = 0.001,
                    slippage_pct: float = 0.0001) -> Tuple[pd.DataFrame, List[Tuple]]:
    """Simulate trades based on generated signals."""
    try:
        capital = initial_capital
        buying_power = initial_capital * (buying_power_pct / 100)
        capital_history = []
        drawdown_info = []
        last_prices = {}

        data.sort_values(['date', 'ticker'], inplace=True)
        grouped = data.groupby('date')
        positions = {}

        for date, group in grouped:
            for idx, row in group.iterrows():
                ticker = row['ticker']
                signal = row['Signal']
                current_price = row['close']

                if current_price <= 0 or pd.isna(current_price):
                    logger.warning(f"Invalid price for {ticker} on {date}. Skipping.")
                    continue

                last_prices[ticker] = current_price

                if ticker not in positions:
                    positions[ticker] = {'position': 0, 'buy_price': 0}

                position = positions[ticker]['position']
                buy_price = positions[ticker]['buy_price']

                # Take-profit logic
                if position > 0 and current_price > buy_price * (1 + take_profit_pct):
                    sell_price = current_price * (1 - slippage_pct)
                    net_proceeds = position * sell_price * (1 - transaction_cost_pct)
                    capital += net_proceeds
                    buying_power += net_proceeds
                    logger.info(f"Take-profit sold {position} shares of {ticker} on {date} at {sell_price:.2f} (net: {net_proceeds:.2f}).")
                    positions[ticker]['position'] = 0
                    positions[ticker]['buy_price'] = 0

                # Stop-loss logic
                if position > 0 and current_price < buy_price * (1 - stop_loss_pct):
                    sell_price = current_price * (1 - slippage_pct)
                    net_proceeds = position * sell_price * (1 - transaction_cost_pct)
                    drawdown_info.append((date, ticker, capital, current_price))
                    capital += net_proceeds
                    buying_power += net_proceeds
                    logger.warning(f"Stop-loss triggered for {ticker} on {date} at {sell_price:.2f}. New capital: {capital:.2f}")
                    positions[ticker]['position'] = 0
                    positions[ticker]['buy_price'] = 0

                # Dynamic risk per trade based on current capital
                risk_per_trade = capital * config.get('risk_per_trade_pct', 0.005) / 100
                atr = row.get('ATR', 1.0)
                position_size = max(1, min(int(risk_per_trade / atr), max_position, int(buying_power / current_price)))
                trade_value = position_size * current_price

                # Buy logic with slippage and costs
                if signal == 1 and positions[ticker]['position'] == 0:
                    buy_price = current_price * (1 + slippage_pct)
                    trade_value = position_size * buy_price * (1 + transaction_cost_pct)
                    if trade_value <= buying_power and trade_value > 0:
                        positions[ticker]['position'] = position_size
                        positions[ticker]['buy_price'] = buy_price
                        capital -= trade_value
                        buying_power -= trade_value
                        logger.info(f"Bought {position_size} shares of {ticker} on {date} at {buy_price:.2f} (cost: {trade_value:.2f}).")
                    else:
                        logger.info(f"Insufficient buying power to execute trade for {ticker} on {date}.")

                # Sell logic with slippage and costs
                elif signal == -1 and positions[ticker]['position'] > 0:
                    sell_price = current_price * (1 - slippage_pct)
                    quantity_to_sell = positions[ticker]['position']
                    trade_value = quantity_to_sell * sell_price * (1 - transaction_cost_pct)
                    capital += trade_value
                    buying_power += trade_value
                    logger.info(f"Sold {quantity_to_sell} shares of {ticker} on {date} at {sell_price:.2f} (net: {trade_value:.2f}).")
                    positions[ticker]['position'] = 0
                    positions[ticker]['buy_price'] = 0

            total_position_value = 0
            for t in positions:
                position_qty = positions[t]['position']
                if position_qty <= 0:
                    continue
                price = None
                subset = data[(data['date'] == date) & (data['ticker'] == t)]
                if not subset.empty:
                    price = subset['close'].values[0]
                if price is None or pd.isna(price) or price <= 0:
                    price = last_prices.get(t)
                if price is None or pd.isna(price) or price <= 0:
                    continue
                total_position_value += position_qty * price
            capital_history.append({'date': date, 'capital': capital + total_position_value})

        capital_history_df = pd.DataFrame(capital_history)
        capital_history_df.set_index('date', inplace=True)
        logger.info("Trade simulation completed successfully.")
        return capital_history_df, drawdown_info
    except Exception as e:
        logger.error(f"Error in simulate_trades: {e}", exc_info=True)
        return pd.DataFrame(), []

def calculate_performance(capital_history: pd.DataFrame, drawdown_info: List[Tuple]):
    """Calculate and print performance metrics."""
    try:
        capital_history['Returns'] = capital_history['capital'].pct_change().fillna(0)
        cumulative_returns = (capital_history['Returns'] + 1).prod() - 1
        sharpe_ratio = (np.mean(capital_history['Returns']) / np.std(capital_history['Returns'])) * np.sqrt(252) if np.std(capital_history['Returns']) != 0 else 0
        max_drawdown = (capital_history['capital'].cummax() - capital_history['capital']).max()

        logger.info(f"Cumulative Returns: {cumulative_returns:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Maximum Drawdown: {max_drawdown:.2f}")

        if drawdown_info:
            logger.info("\nSignificant Drawdown Periods:")
            for date, ticker, cap, price in drawdown_info:
                logger.info(f"Date: {date}, Ticker: {ticker}, Capital: {cap}, Price: {price}")
    except Exception as e:
        logger.error(f"Error in calculate_performance: {e}", exc_info=True)

def log_backtesting_results(capital_history: pd.DataFrame):
    """Log backtesting results into a CSV file."""
    try:
        out_path = os.path.join(ARTIFACT_DIR, 'backtesting_results.csv')
        capital_history.to_csv(out_path)
        logger.debug(f"Backtesting results logged into '{out_path}'.")
        print(f"Backtesting results logged into '{out_path}'.")
    except Exception as e:
        logger.error(f"Failed to log backtesting results: {e}", exc_info=True)

def plot_backtesting_results(capital_history: pd.DataFrame):
    """Plot backtesting results."""
    try:
        capital_history = capital_history.copy()  # Avoid modifying original
        capital_history.index = pd.to_datetime(capital_history.index.str.split('T').str[0])  # Strip time, convert to datetime
        plt.figure(figsize=(12, 6))
        plt.plot(capital_history.index, capital_history['capital'], label='Strategy Capital')
        plt.title('Backtesting Results')
        plt.xlabel('Date')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True)
        out_path = os.path.join(ARTIFACT_DIR, 'backtesting_results.png')
        plt.savefig(out_path)
        plt.close()
        logger.debug(f"Backtesting results plot saved as '{out_path}'.")
        print(f"Backtesting results plot saved as '{out_path}'.")
    except Exception as e:
        logger.error(f"Failed to plot backtesting_results: {e}", exc_info=True)

def send_trade_notification(ticker: str, action: str, price: float, quantity: float, order_type: str, error: str = None):
    """
    Send trade notifications via webhook (Slack/Discord/Telegram/Custom).

    Supports:
    - Slack webhooks
    - Discord webhooks
    - Generic webhooks (any JSON endpoint)
    - Telegram bot API (future enhancement)

    Configuration in config.yaml:
      notifications:
        enabled: true
        webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        # or Discord: "https://discord.com/api/webhooks/YOUR/WEBHOOK/URL"
    """
    try:
        # Check if notifications are enabled
        if not config.get('notifications', {}).get('enabled', False):
            return

        webhook_url = config.get('notifications', {}).get('webhook_url')
        if not webhook_url:
            return

        # Build message
        if error:
            emoji = "❌"
            status_text = f"FAILED: {error}"
            color = "#ff0000"  # Red
        elif order_type == 'limit_filled':
            emoji = "✅"
            status_text = "Limit order filled (saved slippage!)"
            color = "#00ff00"  # Green
        else:
            emoji = "🤖"
            status_text = f"Market order ({order_type})"
            color = "#0099ff"  # Blue

        message_text = f"{emoji} **{action.upper()}** {quantity} {ticker} @ ${price:.2f}\n{status_text}"

        # Detect webhook type and format accordingly
        if 'slack.com' in webhook_url:
            # Slack format
            payload = {
                "text": f"{emoji} Trade Alert",
                "attachments": [{
                    "color": color,
                    "fields": [
                        {"title": "Action", "value": action.upper(), "short": True},
                        {"title": "Ticker", "value": ticker, "short": True},
                        {"title": "Quantity", "value": str(quantity), "short": True},
                        {"title": "Price", "value": f"${price:.2f}", "short": True},
                        {"title": "Type", "value": order_type, "short": False}
                    ]
                }]
            }
        elif 'discord.com' in webhook_url:
            # Discord format
            payload = {
                "content": message_text,
                "embeds": [{
                    "title": f"{action.upper()} {ticker}",
                    "description": f"Quantity: {quantity}\nPrice: ${price:.2f}\nType: {order_type}",
                    "color": int(color.replace('#', ''), 16) if color.startswith('#') else 255
                }]
            }
        else:
            # Generic webhook format
            payload = {
                "text": message_text,
                "ticker": ticker,
                "action": action,
                "quantity": quantity,
                "price": price,
                "order_type": order_type,
                "error": error
            }

        # Send notification
        import requests
        response = requests.post(webhook_url, json=payload, timeout=5)
        if response.status_code != 200:
            logger.warning(f"Notification webhook returned {response.status_code}")

    except Exception as e:
        # Don't let notification failures break trading
        logger.debug(f"Failed to send trade notification: {e}")

def backtest_strategy(model: XGBClassifier, data: pd.DataFrame, selected_features: List[str], config, q_table, initial_capital: float = 10000):
    """Backtest the trading strategy on historical data."""
    logger.info("Starting backtesting on historical data.")
    # Avoid re-calling sentiment if it is already present (GUI backtest/training pipeline may have added it).
    if 'Sentiment_Score' not in data.columns:
        data = add_sentiment_features(data, config)
    data = generate_signals(model, data, selected_features, config, q_table, threshold=0.5, positions_snapshot={})

    if data.empty:
        logger.error("No data available after signal generation for backtesting. Exiting.")
        return

    capital_history, drawdown_info = simulate_trades(data, initial_capital, config=config)
    calculate_performance(capital_history, drawdown_info)
    log_backtesting_results(capital_history)
    plot_backtesting_results(capital_history)
    logger.info("Backtesting process completed successfully.")

def place_order(api: tradeapi.REST, symbol: str, qty: float, side: str, order_type: str = 'market', time_in_force: str = 'day', current_price: float = None, use_smart_order: bool = True):
    """
    Place an order through Alpaca API with smart execution.

    Smart order logic:
    1. Try limit order at slightly better price (0.1% improvement)
    2. If not filled immediately (IOC), fallback to market order
    3. This saves on slippage while ensuring execution

    Args:
        api: Alpaca REST API client
        symbol: Ticker symbol
        qty: Quantity to trade
        side: 'buy' or 'sell'
        order_type: 'market', 'limit', or 'smart' (default: 'market')
        time_in_force: 'day', 'gtc', 'ioc', etc.
        current_price: Current market price (required for smart orders)
        use_smart_order: Whether to use smart limit+market fallback (default: True)
    """
    try:
        # Smart order: Try limit first, fallback to market
        if use_smart_order and current_price and order_type in ['market', 'smart']:
            try:
                # Calculate limit price (0.1% better than market)
                if side == 'buy':
                    limit_price = round(current_price * 0.999, 2)  # 0.1% below market
                else:
                    limit_price = round(current_price * 1.001, 2)  # 0.1% above market

                logger.info(f"Attempting smart order for {symbol}: limit @ ${limit_price:.2f} (market: ${current_price:.2f})")

                # Try limit order with IOC (immediate-or-cancel)
                limit_order = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='limit',
                    limit_price=limit_price,
                    time_in_force='ioc'  # Immediate or cancel
                )

                # Check if filled immediately
                import time
                time.sleep(1)  # Wait 1 second for fill
                order_status = api.get_order(limit_order.id)

                if order_status.status in ['filled', 'partially_filled']:
                    logger.info(f"✓ Limit order filled for {symbol} @ ${limit_price:.2f}")
                    send_trade_notification(symbol, side, limit_price, qty, 'limit_filled')
                    return limit_order
                else:
                    logger.info(f"Limit order not filled, falling back to market order for {symbol}")
                    # Cancel the limit order
                    try:
                        api.cancel_order(limit_order.id)
                    except:
                        pass  # Already cancelled or filled

            except Exception as e:
                logger.debug(f"Smart limit order failed for {symbol}, using market: {e}")

        # Standard market order (or fallback from failed limit)
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market' if order_type in ['market', 'smart'] else order_type,
            time_in_force=time_in_force
        )
        logger.info(f"Order placed: {side.upper()} {qty} {symbol} @ market")
        send_trade_notification(symbol, side, current_price or 0, qty, 'market')
        return order

    except Exception as e:
        logger.error(f"Error placing order for {symbol}: {e}", exc_info=True)
        send_trade_notification(symbol, side, current_price or 0, qty, 'failed', error=str(e))
        return None

def monitor_order(api: tradeapi.REST, order_id: str, max_attempts: int = 10, wait_time: int = 5):
    """Monitor the status of an order."""
    attempts = 0
    while attempts < max_attempts:
        try:
            order = api.get_order(order_id)
            if order.status == 'filled':
                logger.info(f"Order {order_id} filled.")
                return True
            elif order.status in ['canceled', 'rejected', 'expired']:
                logger.warning(f"Order {order_id} status: {order.status}. Aborting.")
                return False
            else:
                logger.info(f"Order {order_id} status: {order.status}. Waiting...")
                time.sleep(wait_time)
                attempts += 1
        except Exception as e:
            logger.error(f"Error monitoring order {order_id}: {e}", exc_info=True)
            return False
    logger.warning(f"Order {order_id} not filled after {max_attempts} attempts.")
    return False

def execute_trading_logic_live(api: tradeapi.REST, data: pd.DataFrame, config, q_table, guard: DayTradeGuard, dexter_gate: DexterGate, buying_power_pct: float = 100):
    """Execute trading logic using hybrid signals during live trading with PDT guard and Dexter veto."""
    if data.empty:
        logger.warning("No data available for executing trading logic. Skipping trading.")
        return

    # Safety check: Don't trade if too close to market close (after 3:45 PM ET)
    now_et = datetime.now(tz=pytz.timezone('US/Eastern'))
    cutoff_time = dt_time(15, 45)  # 3:45 PM ET
    if now_et.time() >= cutoff_time and now_et.weekday() < 5:
        logger.warning(f"Too close to market close ({now_et.strftime('%H:%M:%S')} ET). Aborting trading to prevent unfilled orders.")
        return

    try:
        data.sort_values(['date', 'ticker'], inplace=True)
    except KeyError as e:
        logger.error(f"DataFrame missing expected columns: {e}. Skipping trading.")
        return

    account = api.get_account()
    available_cash = float(account.cash)
    buying_power = float(account.buying_power) * (buying_power_pct / 100)
    log_filename = initialize_trade_log()
    logger.info(f"Starting live trading at {now_et.strftime('%H:%M:%S')} ET with cash: {available_cash} and buying power cap: {buying_power}")

    current_positions = fetch_current_positions(api)
    positions_dict = {pos['ticker']: pos for pos in current_positions}
    entry_dates = {pos['ticker']: datetime.now().strftime('%Y-%m-%d') for pos in current_positions}

    trade_count = 0
    successful_trades = 0

    for idx, row in data.iterrows():
        ticker = row['ticker']
        signal = row['Signal']
        current_price = row['close']
        date = row['date']
        asset_config = get_asset_config(ticker, config)
        is_crypto = is_crypto_ticker(ticker, config)

        atr = row.get('ATR', 1.0)
        if np.isnan(atr) or atr < 0.01:
            logger.warning(f"ATR is NaN or too small for {ticker} on {date}. Skipping trade.")
            continue

        # Grok 4.20 Enhancement: Volatility regime-based position sizing
        vol_regime = row.get('Vol_Regime', 1)  # 0=low, 1=medium, 2=high
        vol_percentile = row.get('Vol_Percentile', 0.5)

        # Position size multiplier based on volatility regime
        # Low vol: increase size (more opportunity, less risk)
        # High vol: decrease size (preserve capital)
        vol_size_multiplier = {0: 1.3, 1: 1.0, 2: 0.7}.get(int(vol_regime), 1.0)

        now_dt = datetime.now(tz=pytz.timezone('US/Eastern'))
        guard_remaining = guard.remaining_today(now_dt) if not is_crypto else 999
        risk_per_trade_pct = asset_config.get('risk_per_trade_pct', config.get('risk_per_trade_pct', 0.01))
        risk_multiplier = asset_config.get('risk_multiplier', asset_config.get('crypto_risk_multiplier', 1.0))

        # Apply volatility adjustment to risk
        adjusted_risk_multiplier = float(risk_multiplier) * vol_size_multiplier
        risk_per_trade = available_cash * (float(risk_per_trade_pct) * adjusted_risk_multiplier) / 100

        max_position_pct = asset_config.get('max_position_pct', config.get('max_position_pct', 5.0))
        # Also adjust max position by volatility (tighter limits in high vol)
        adjusted_max_position_pct = float(max_position_pct) * vol_size_multiplier

        equity = float(getattr(account, "equity", available_cash))
        max_position_value = equity * (adjusted_max_position_pct / 100)
        position_size = min((risk_per_trade / atr), (buying_power / current_price))
        if not is_crypto:
            position_size = max(1, int(position_size))
        if max_position_value > 0:
            position_size = min(position_size, (max_position_value / current_price))
        if not is_crypto:
            position_size = int(position_size)
        if position_size <= 0:
            logger.info(f"Position size too small for {ticker} on {date}. Skipping trade.")
            continue
        trade_value = position_size * current_price

        # Grok 4.20 Enhancement: Dynamic stop loss and take profit levels
        # Base levels from config, adjusted by ATR and volatility
        base_stop_loss_pct = asset_config.get('stop_loss_pct', 0.05 if not is_crypto else 0.15)
        base_take_profit_pct = asset_config.get('take_profit_pct', 0.10 if not is_crypto else 0.25)

        # ATR-based adjustment: wider stops in volatile conditions
        atr_pct = atr / current_price if current_price > 0 else 0.02
        # Stop loss: at least 2x ATR, scaled by vol regime
        vol_stop_multiplier = {0: 1.5, 1: 2.0, 2: 2.5}.get(int(vol_regime), 2.0)
        dynamic_stop_loss_pct = max(base_stop_loss_pct, atr_pct * vol_stop_multiplier)
        dynamic_stop_loss_pct = min(dynamic_stop_loss_pct, 0.20)  # Cap at 20%

        # Take profit: at least 2x stop loss (reward/risk ratio)
        dynamic_take_profit_pct = max(base_take_profit_pct, dynamic_stop_loss_pct * 2)
        dynamic_take_profit_pct = min(dynamic_take_profit_pct, 0.30)  # Cap at 30%

        logger.debug(f"{ticker}: Vol regime={vol_regime}, Size multiplier={vol_size_multiplier:.2f}, "
                    f"Stop={dynamic_stop_loss_pct:.2%}, TP={dynamic_take_profit_pct:.2%}")

        logger.debug(f"Ticker: {ticker}, Date: {date}, Signal: {signal}, Position size: {position_size}, Trade value: {trade_value}, Buying power: {buying_power}")

        if guard_remaining <= 0 and signal == 1 and not is_crypto:
            logger.info(f"PDT guard blocking new entry for {ticker}; remaining day trades: {guard_remaining}")
            continue

        dexter_gate.refresh()
        if signal != 0:
            allowed = dexter_gate.should_allow(ticker, guard_remaining, {
                'signal': signal,
                'price': current_price,
                'atr': atr,
                'sentiment': row.get('Sentiment_Score', 0),
            })
            if not allowed:
                logger.info(f"Dexter gate blocked trade for {ticker}.")
                continue

        # Signal Confidence Scoring - scale position based on signal agreement
        confidence_cfg = config.get('confidence', {})
        if confidence_cfg.get('enabled', True) and signal != 0:
            # Get Dexter bias for this ticker
            dexter_bias = dexter_gate.get_bias(ticker) if hasattr(dexter_gate, 'get_bias') else None

            # Get sentiment score from row
            sentiment_score = row.get('Sentiment_Score', None)

            # Get technical signal from row
            tech_signal = 'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'

            # Get ML prediction (signal is already from ML)
            ml_pred = 1 if signal == 1 else 0

            # Get FinViz risk from Dexter bias enrichment
            finviz_risk = None
            if dexter_bias and 'fundamentals' in dexter_bias:
                finviz_risk = dexter_bias.get('fundamentals', {})

            # Calculate confidence
            confidence = calculate_confidence(
                ticker=ticker,
                dexter_bias=dexter_bias,
                sentiment_score=sentiment_score,
                technical_signal=tech_signal,
                ml_prediction=ml_pred,
                finviz_risk=finviz_risk,
                config=config
            )

            # Apply position multiplier based on confidence
            if confidence.position_multiplier == 0.0:
                logger.info(f"[{ticker}] Confidence too low ({confidence.score:.0%}), skipping trade. Signals disagreed.")
                continue

            # Scale position size by confidence
            original_position_size = position_size
            position_size = position_size * confidence.position_multiplier
            if not is_crypto:
                position_size = max(1, int(position_size))

            if position_size != original_position_size:
                logger.info(f"[{ticker}] Position scaled by confidence: {original_position_size} -> {position_size} "
                           f"({confidence.position_multiplier:.0%} of full, confidence={confidence.score:.0%})")

            # Update trade value after scaling
            trade_value = position_size * current_price

        if trade_value > buying_power:
            logger.info(f"Insufficient buying power to execute trade for {ticker} on {date}.")
            continue
        time_in_force = asset_config.get('time_in_force', 'gtc' if is_crypto else 'day')

        if signal == 1:
            if ticker not in positions_dict or positions_dict[ticker]['quantity'] == 0:
                qty = float(position_size)
                if not is_crypto:
                    qty = int(qty)
                min_qty = float(asset_config.get('min_qty', 0.0001)) if is_crypto else 1
                if qty < min_qty:
                    logger.info(f"Calculated quantity {qty} below minimum for {ticker}. Skipping.")
                    continue

                # Grok 4.20 Enhancement: Use bracket order (OCO) for automatic stop loss and take profit
                # Calculate stop loss and take profit prices
                stop_loss_price = round(current_price * (1 - dynamic_stop_loss_pct), 2)
                take_profit_price = round(current_price * (1 + dynamic_take_profit_pct), 2)

                use_bracket = config.get('use_bracket_orders', True) and not is_crypto

                if use_bracket:
                    try:
                        # Alpaca bracket order: entry + stop loss + take profit
                        logger.info(f"Placing bracket order for {ticker}: Entry={current_price:.2f}, "
                                   f"Stop={stop_loss_price:.2f} ({dynamic_stop_loss_pct:.1%}), "
                                   f"TP={take_profit_price:.2f} ({dynamic_take_profit_pct:.1%})")
                        order = api.submit_order(
                            symbol=ticker,
                            qty=qty,
                            side='buy',
                            type='market',
                            time_in_force=time_in_force,
                            order_class='bracket',
                            stop_loss={'stop_price': stop_loss_price},
                            take_profit={'limit_price': take_profit_price}
                        )
                        logger.info(f"✓ Bracket order submitted for {ticker}: {order.id}")
                    except Exception as bracket_error:
                        logger.warning(f"Bracket order failed for {ticker}: {bracket_error}. Falling back to simple order.")
                        order = place_order(api, ticker, qty, 'buy', time_in_force=time_in_force, current_price=current_price)
                else:
                    order = place_order(api, ticker, qty, 'buy', time_in_force=time_in_force, current_price=current_price)

                if order:
                    buying_power -= trade_value
                    available_cash -= trade_value
                    trade_count += 1
                    log_trade({
                        'ticker': ticker,
                        'type': 'buy',
                        'price': current_price,
                        'quantity': position_size,
                        'stop_loss': stop_loss_price if use_bracket else None,
                        'take_profit': take_profit_price if use_bracket else None,
                        'status': 'submitted'
                    }, log_filename)
                    if monitor_order(api, order.id):
                        successful_trades += 1
                        if not is_crypto:
                            guard.record_entry(ticker, now_dt)
                        entry_dates[ticker] = date
                        positions_dict[ticker] = {
                            'quantity': qty,
                            'entry_price': current_price,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price
                        }
        elif signal == -1:
            if ticker in positions_dict and positions_dict[ticker]['quantity'] > 0:
                qty = float(position_size)
                if not is_crypto:
                    qty = int(qty)
                quantity_to_sell = min(positions_dict[ticker]['quantity'], qty)
                trade_value = quantity_to_sell * current_price
                order = place_order(api, ticker, quantity_to_sell, 'sell', time_in_force=time_in_force, current_price=current_price)
                if order:
                    buying_power += trade_value
                    available_cash += trade_value
                    trade_count += 1
                    log_trade({
                        'ticker': ticker,
                        'type': 'sell',
                        'price': current_price,
                        'quantity': quantity_to_sell,
                        'status': 'submitted'
                    }, log_filename)
                    if monitor_order(api, order.id):
                        successful_trades += 1
                        if not is_crypto:
                            guard.record_exit(ticker, now_dt)
                        positions_dict[ticker]['quantity'] -= quantity_to_sell
                        if positions_dict[ticker]['quantity'] == 0:
                            del positions_dict[ticker]

    success_rate = (successful_trades / trade_count) * 100 if trade_count > 0 else 0
    with open(log_filename, mode='a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Summary", "", "Total Trades", trade_count, "Successful Trades", successful_trades, "Success Rate", f"{success_rate:.2f}%"])
    logger.info(f"Live trading session completed. Total trades: {trade_count}, Successful trades: {successful_trades}, Success rate: {success_rate:.2f}%")

# =============================================================================
# Continuous Trading Loop (Hybrid Execution)
# =============================================================================

def run_continuous_trading(api: tradeapi.REST, model: XGBClassifier, config, q_table, selected_features: List[str], guard: DayTradeGuard, dexter_gate: DexterGate):
    """Run the trading bot once per day at 2:30 PM ET to ensure orders complete before 4:00 PM market close."""
    logger.info("Starting daily close trading mode (executes at 2:30 PM ET)...")
    logger.info("90-minute buffer before close ensures sentiment analysis and order placement complete before 4:00 PM.")
    raw_historical = None
    job_lock = threading.Lock()  # Thread safety lock

    def job():
        with job_lock:
            nonlocal raw_historical
            current_time = datetime.now(tz=pytz.timezone('US/Eastern'))
            logger.info(f"Job running at {current_time.strftime('%H:%M:%S')} ET")

            # Check if we're within the execution window (2:30 PM - 3:30 PM)
            # This gives 30-90 minute buffer before 4:00 PM close
            now_time = current_time.time()
            execution_start = dt_time(14, 30)   # 2:30 PM
            execution_end = dt_time(15, 30)    # 3:30 PM

            if execution_start <= now_time <= execution_end and current_time.weekday() < 5:
                logger.info("Market close window (2:30-3:30 PM). Proceeding with trading logic...")
                # Fetch raw historical data once or update incrementally
                if raw_historical is None:
                    raw_historical = fetch_current_market_data_with_crypto(
                        config['tickers'],
                        config['polygon']['api_key'],
                        api,
                        config=config,
                    )
                    if raw_historical.empty:
                        logger.error("Failed to fetch initial historical data. Will retry tomorrow...")
                        return

                # Fetch latest data (should be nearly complete daily bar at 3:55 PM)
                latest_data = fetch_latest_data(config['tickers'], api, config=config)
                if not latest_data.empty:
                    # Append latest data to raw historical data
                    raw_historical = pd.concat([raw_historical, latest_data]).drop_duplicates(subset=['ticker', 'date'], keep='last')
                    raw_historical = raw_historical.groupby('ticker').tail(200).reset_index(drop=True)  # Keep last 200 for buffer
                    logger.debug(f"Combined raw data rows after tail: {len(raw_historical)}")
                    for ticker in config['tickers']:
                        if ticker != '^VIX':
                            rows = len(raw_historical[raw_historical['ticker'] == ticker])
                            logger.debug(f"Rows for {ticker} after tail: {rows}")
                    engineered_data = engineer_features(raw_historical)
                    if engineered_data.empty:
                        logger.warning("Feature engineering resulted in empty DataFrame. Skipping this cycle.")
                        return
                    # Add sentiment with timeout protection (max 5 minutes)
                    try:
                        import signal
                        def timeout_handler(signum, frame):
                            raise TimeoutError("Sentiment analysis exceeded 5 minute timeout")

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(300)  # 5 minute timeout

                        engineered_data = add_sentiment_features(engineered_data, config)
                        signal.alarm(0)  # Cancel alarm if completed
                        logger.info("Sentiment analysis completed successfully")
                    except (Exception, TimeoutError) as e:
                        signal.alarm(0)  # Cancel alarm
                        logger.error(f"Sentiment analysis failed or timed out: {e}. Using zero sentiment as fallback.")
                        engineered_data['Sentiment_Score'] = 0.0
                    if not engineered_data.empty:
                        positions_snapshot = {p['ticker']: p for p in fetch_current_positions(api)}
                        signals = generate_signals(model, engineered_data, selected_features, config, q_table, positions_snapshot=positions_snapshot)
                        if not signals.empty:
                            execute_trading_logic_live(api, signals, config, q_table, guard, dexter_gate)
                else:
                    logger.warning("No new data fetched. Skipping this cycle.")
            else:
                logger.info("Outside execution window (2:30-3:30 PM ET). Skipping job.")

    # Run immediately at startup if within execution window
    current_time = datetime.now(tz=pytz.timezone('US/Eastern'))
    now_time = current_time.time()
    execution_start = dt_time(14, 30)   # 2:30 PM
    execution_end = dt_time(15, 30)    # 3:30 PM

    if execution_start <= now_time <= execution_end and current_time.weekday() < 5:
        logger.info("Within execution window at startup. Running initial job...")
        job()
    else:
        logger.info("Outside execution window at startup. Waiting for 2:30 PM ET...")

    # Schedule job to run daily at 2:30 PM ET (90 minutes before market close)
    execution_time = "14:30"  # 2:30 PM ET
    schedule.every().day.at(execution_time).do(job)
    logger.info(f"Scheduled daily execution at {execution_time} ET (90 min before close)")

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute instead of every 5 seconds

# =============================================================================
# Main Execution
# =============================================================================

def main():
    global config
    print("DEBUG: Entering main()", flush=True)
    config_file = "config.yaml"
    print("DEBUG: Loading configuration...", flush=True)
    config = load_configuration(config_file)
    print("DEBUG: Configuration loaded", flush=True)
    maybe_fetch_tickers_via_dexter(config)
    print("DEBUG: Dexter fetch completed", flush=True)
    config = maybe_override_tickers_from_json(config)
    print("DEBUG: Ticker override completed", flush=True)
    config = maybe_add_crypto_tickers(config)
    print("DEBUG: Crypto tickers added", flush=True)

    api_key = config['alpaca']['api_key']
    api_secret = config['alpaca']['api_secret']
    base_url = config['alpaca']['base_url']
    tickers = config['tickers']
    print(f"DEBUG: Got {len(tickers)} tickers: {tickers}", flush=True)

    # Combined selected features (core features always available)
    selected_features = [
        'MA10', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'Bollinger_Upper',
        'Bollinger_Lower', 'Lag1_Close', 'Lag2_Close', 'ATR', 'Stochastic_RSI', 'Volume_Change'
    ]  # Exclude Sentiment_Score initially

    # Add TA-Lib features only if available
    if talib:
        selected_features.extend(['Momentum', 'SMA_20'])
        logger.info("TA-Lib is available. Adding Momentum and SMA_20 features.")
    else:
        logger.info("TA-Lib not available. Using core features only.")

    # Add FinViz fundamental features if available
    try:
        from finviz_enrichment import FINVIZ_AVAILABLE
        if FINVIZ_AVAILABLE:
            fundamental_features = [
                'PE_Ratio', 'Forward_PE', 'PEG_Ratio', 'Debt_Equity',
                'ROE', 'Profit_Margin', 'Short_Float', 'Beta',
                'Analyst_Recom', 'SMA20_Dist', 'SMA50_Dist'
            ]
            selected_features.extend(fundamental_features)
            logger.info(f"FinViz available. Adding {len(fundamental_features)} fundamental features.")
    except ImportError:
        logger.info("FinViz not available. Skipping fundamental features.")

    print("DEBUG: Initializing Alpaca API...", flush=True)
    api = initialize_alpaca_api(api_key, api_secret, base_url)
    print("DEBUG: Alpaca API initialized", flush=True)

    print("DEBUG: Getting account details...", flush=True)
    account = api.get_account()
    print("DEBUG: Account details retrieved", flush=True)
    logger.info(f"Account Details - Cash: {account.cash}, Buying Power: {account.buying_power}")
    print("DEBUG: Creating guards...", flush=True)
    guard = DayTradeGuard(config.get('max_day_trades', 2))
    dexter_gate = DexterGate()
    print("DEBUG: Guards created", flush=True)

    # Check for environment variable or command-line argument to skip interactive prompt
    print("DEBUG: Checking retrain argument...", flush=True)
    if os.environ.get('RETRAIN_MODEL', '').lower() == 'yes':
        trained_today = 'no'  # Force retraining
    elif os.environ.get('RETRAIN_MODEL', '').lower() == 'no':
        trained_today = 'yes'  # Skip retraining
    elif len(sys.argv) > 1 and sys.argv[1] in ['--retrain', '--no-retrain']:
        trained_today = 'no' if sys.argv[1] == '--retrain' else 'yes'
    else:
        # Interactive mode
        try:
            trained_today = input("Have you downloaded, trained, and backtested the model today? (yes/no): ").strip().lower()
        except EOFError:
            # Non-interactive environment, default to retraining with crypto
            logger.warning("Non-interactive environment detected. Defaulting to model retraining.")
            trained_today = 'no'
    print(f"DEBUG: Retrain decision: {trained_today}", flush=True)

    if trained_today == "no":
        logger.info("Starting data download, model training, and backtesting...")
        print("DEBUG: Starting data download...", flush=True)
        download_historical_data_with_crypto(tickers, config)
        print("DEBUG: Data download completed", flush=True)
        data = load_historical_data("./data", config)
        if data.empty:
            logger.error("No data loaded. Exiting.")
            sys.exit(1)

        data = engineer_features(data)
        data = add_sentiment_features(data, config)
        if data.empty:
            logger.error("No data after feature engineering. Exiting.")
            sys.exit(1)

        selected_features_with_sentiment = selected_features + ['Sentiment_Score']  # Add after sentiment
        X_train, X_test, y_train, y_test = prepare_train_test_data(data, selected_features_with_sentiment)
        model = tune_and_train_model(X_train, y_train)
        if model:
            evaluate_model(model, X_test, y_test)
            q_table = load_q_table()
            backtest_strategy(model, data, selected_features_with_sentiment, config, q_table)

    elif trained_today == "yes":
        logger.info("Skipping data download, training, and backtesting as requested.")
    else:
        logger.error("Invalid input. Please enter 'yes' or 'no'. Exiting.")
        sys.exit(1)

    logger.info("Starting continuous trading...")
    try:
        model_path = os.path.join(ARTIFACT_DIR, 'final_model.pkl')
        if not os.path.exists(model_path):
            logger.error("Model file 'artifacts/final_model.pkl' not found. Please train the model first.")
            sys.exit(1)
        model = joblib.load(model_path)
        logger.info("Loaded pre-trained model from 'artifacts/final_model.pkl'.")

        # Dynamically detect features the model was trained with
        model_features = model.get_booster().feature_names
        logger.info(f"Model was trained with features: {model_features}")

        q_table = load_q_table()
        run_continuous_trading(api, model, config, q_table, model_features, guard, dexter_gate)
    except Exception as e:
        logger.error(f"Error during continuous trading: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Trading bot interrupted by user. Exiting gracefully.")
    except Exception as e:
        logger.error(f"Unhandled exception in main execution: {e}", exc_info=True)
        sys.exit(1)
