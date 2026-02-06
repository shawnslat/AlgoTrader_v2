"""
FinViz Enrichment Module

Provides fundamental data from FinViz to:
1. Enrich dexter_bias.json with additional risk flags
2. Add fundamental features for ML model training

Uses finvizfinance package: https://github.com/lit26/finvizfinance
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

try:
    from finvizfinance.quote import finvizfinance
    FINVIZ_AVAILABLE = True
except ImportError:
    FINVIZ_AVAILABLE = False

logger = logging.getLogger(__name__)


# Risk thresholds for bias enrichment
RISK_THRESHOLDS = {
    'pe_extreme_high': 100,       # P/E > 100 is risky
    'pe_negative': 0,             # Negative P/E (losing money)
    'insider_sell_pct': -5,       # >5% insider selling in recent period
    'short_float_high': 20,       # >20% short float is risky
    'volatility_high': 5,         # >5% weekly volatility
    'analyst_strong_sell_pct': 30, # >30% analysts say sell
    'debt_equity_high': 2,        # D/E > 2 is concerning
    'rsi_overbought': 70,         # RSI > 70 overbought
    'rsi_oversold': 30,           # RSI < 30 oversold
}


def get_stock_fundamentals(ticker: str, retry_count: int = 2) -> Optional[Dict[str, Any]]:
    """
    Fetch fundamental data for a single ticker from FinViz.

    Returns dict with parsed fundamentals or None on failure.
    """
    if not FINVIZ_AVAILABLE:
        logger.warning("finvizfinance not installed. Run: pip install finvizfinance")
        return None

    for attempt in range(retry_count):
        try:
            stock = finvizfinance(ticker)
            fundament = stock.ticker_fundament()

            # Parse key metrics (FinViz returns strings, need to convert)
            parsed = {
                'ticker': ticker,
                'fetched_at': datetime.now().isoformat(),
                'company': fundament.get('Company', ''),
                'sector': fundament.get('Sector', ''),
                'industry': fundament.get('Industry', ''),
                'country': fundament.get('Country', ''),
                'market_cap': _parse_market_cap(fundament.get('Market Cap', '')),
                'pe': _parse_float(fundament.get('P/E', '')),
                'forward_pe': _parse_float(fundament.get('Forward P/E', '')),
                'peg': _parse_float(fundament.get('PEG', '')),
                'ps': _parse_float(fundament.get('P/S', '')),
                'pb': _parse_float(fundament.get('P/B', '')),
                'debt_equity': _parse_float(fundament.get('Debt/Eq', '')),
                'roe': _parse_percent(fundament.get('ROE', '')),
                'roi': _parse_percent(fundament.get('ROI', '')),
                'gross_margin': _parse_percent(fundament.get('Gross Margin', '')),
                'oper_margin': _parse_percent(fundament.get('Oper. Margin', '')),
                'profit_margin': _parse_percent(fundament.get('Profit Margin', '')),
                'eps': _parse_float(fundament.get('EPS (ttm)', '')),
                'eps_growth_yoy': _parse_percent(fundament.get('EPS Q/Q', '')),
                'sales_growth_yoy': _parse_percent(fundament.get('Sales Q/Q', '')),
                'insider_own': _parse_percent(fundament.get('Insider Own', '')),
                'insider_trans': _parse_percent(fundament.get('Insider Trans', '')),
                'inst_own': _parse_percent(fundament.get('Inst Own', '')),
                'inst_trans': _parse_percent(fundament.get('Inst Trans', '')),
                'short_float': _parse_percent(fundament.get('Short Float', '')),
                'short_ratio': _parse_float(fundament.get('Short Ratio', '')),
                'target_price': _parse_float(fundament.get('Target Price', '')),
                'rsi_14': _parse_float(fundament.get('RSI (14)', '')),
                'volatility_week': _parse_percent(fundament.get('Volatility', '').split()[0] if fundament.get('Volatility') else ''),
                'volatility_month': _parse_percent(fundament.get('Volatility', '').split()[1] if len(fundament.get('Volatility', '').split()) > 1 else ''),
                'beta': _parse_float(fundament.get('Beta', '')),
                'atr': _parse_float(fundament.get('ATR', '')),
                'sma20': _parse_percent(fundament.get('SMA20', '')),
                'sma50': _parse_percent(fundament.get('SMA50', '')),
                'sma200': _parse_percent(fundament.get('SMA200', '')),
                '52w_high': _parse_percent(fundament.get('52W High', '')),
                '52w_low': _parse_percent(fundament.get('52W Low', '')),
                'avg_volume': _parse_volume(fundament.get('Avg Volume', '')),
                'price': _parse_float(fundament.get('Price', '')),
                'change': _parse_percent(fundament.get('Change', '')),
                'earnings_date': fundament.get('Earnings', ''),
                'recommendation': _parse_float(fundament.get('Recom', '')),  # 1=Buy, 5=Sell
            }

            return parsed

        except Exception as e:
            logger.warning(f"FinViz fetch failed for {ticker} (attempt {attempt + 1}): {e}")
            if attempt < retry_count - 1:
                time.sleep(1)  # Rate limit protection

    return None


def get_stock_news(ticker: str, limit: int = 5) -> List[Dict[str, str]]:
    """Fetch recent news headlines for a ticker."""
    if not FINVIZ_AVAILABLE:
        return []

    try:
        stock = finvizfinance(ticker)
        news_df = stock.ticker_news()

        if news_df is None or news_df.empty:
            return []

        news_list = []
        for _, row in news_df.head(limit).iterrows():
            news_list.append({
                'date': str(row.get('Date', '')),
                'title': row.get('Title', ''),
                'link': row.get('Link', ''),
                'source': row.get('Source', ''),
            })

        return news_list

    except Exception as e:
        logger.warning(f"FinViz news fetch failed for {ticker}: {e}")
        return []


def get_insider_activity(ticker: str) -> List[Dict[str, Any]]:
    """Fetch recent insider trading activity."""
    if not FINVIZ_AVAILABLE:
        return []

    try:
        stock = finvizfinance(ticker)
        insider_df = stock.ticker_inside_trader()

        if insider_df is None or insider_df.empty:
            return []

        insider_list = []
        for _, row in insider_df.head(10).iterrows():
            insider_list.append({
                'owner': row.get('Insider Trading', ''),
                'relationship': row.get('Relationship', ''),
                'date': str(row.get('Date', '')),
                'transaction': row.get('Transaction', ''),
                'cost': row.get('Cost', ''),
                'shares': row.get('#Shares', ''),
                'value': row.get('Value ($)', ''),
                'total_shares': row.get('#Shares Total', ''),
            })

        return insider_list

    except Exception as e:
        logger.warning(f"FinViz insider fetch failed for {ticker}: {e}")
        return []


def assess_fundamental_risk(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze fundamentals and return risk assessment.

    Returns dict with:
        - risk_score: 0-100 (higher = riskier)
        - risk_flags: list of specific concerns
        - recommendation: 'ok', 'caution', or 'avoid'
    """
    if not fundamentals:
        return {'risk_score': 50, 'risk_flags': ['no_data'], 'recommendation': 'caution'}

    risk_flags = []
    risk_score = 0

    # P/E Analysis
    pe = fundamentals.get('pe')
    if pe is not None:
        if pe < 0:
            risk_flags.append(f'negative_pe ({pe:.1f})')
            risk_score += 15
        elif pe > RISK_THRESHOLDS['pe_extreme_high']:
            risk_flags.append(f'extreme_pe ({pe:.1f})')
            risk_score += 10

    # Insider Selling
    insider_trans = fundamentals.get('insider_trans')
    if insider_trans is not None and insider_trans < RISK_THRESHOLDS['insider_sell_pct']:
        risk_flags.append(f'insider_selling ({insider_trans:.1f}%)')
        risk_score += 20

    # Short Interest
    short_float = fundamentals.get('short_float')
    if short_float is not None and short_float > RISK_THRESHOLDS['short_float_high']:
        risk_flags.append(f'high_short_interest ({short_float:.1f}%)')
        risk_score += 15

    # Volatility
    vol_week = fundamentals.get('volatility_week')
    if vol_week is not None and vol_week > RISK_THRESHOLDS['volatility_high']:
        risk_flags.append(f'high_volatility ({vol_week:.1f}%)')
        risk_score += 10

    # Debt/Equity
    debt_eq = fundamentals.get('debt_equity')
    if debt_eq is not None and debt_eq > RISK_THRESHOLDS['debt_equity_high']:
        risk_flags.append(f'high_debt ({debt_eq:.1f}x)')
        risk_score += 10

    # RSI Extremes
    rsi = fundamentals.get('rsi_14')
    if rsi is not None:
        if rsi > RISK_THRESHOLDS['rsi_overbought']:
            risk_flags.append(f'overbought_rsi ({rsi:.0f})')
            risk_score += 5
        elif rsi < RISK_THRESHOLDS['rsi_oversold']:
            risk_flags.append(f'oversold_rsi ({rsi:.0f})')
            # Oversold might be opportunity, not necessarily risk
            risk_score += 2

    # Analyst Recommendation (1=Strong Buy, 5=Strong Sell)
    recom = fundamentals.get('recommendation')
    if recom is not None and recom >= 4:
        risk_flags.append(f'analyst_bearish ({recom:.1f})')
        risk_score += 15

    # 52-week positioning (near highs can be risky for entries)
    high_52w = fundamentals.get('52w_high')
    if high_52w is not None and high_52w > -5:  # Within 5% of 52w high
        risk_flags.append(f'near_52w_high ({high_52w:.1f}%)')
        risk_score += 5

    # Negative margins
    profit_margin = fundamentals.get('profit_margin')
    if profit_margin is not None and profit_margin < 0:
        risk_flags.append(f'negative_margin ({profit_margin:.1f}%)')
        risk_score += 10

    # Cap risk score at 100
    risk_score = min(risk_score, 100)

    # Determine recommendation
    if risk_score >= 40:
        recommendation = 'avoid'
    elif risk_score >= 20:
        recommendation = 'caution'
    else:
        recommendation = 'ok'

    return {
        'risk_score': risk_score,
        'risk_flags': risk_flags,
        'recommendation': recommendation,
    }


def enrich_dexter_bias(bias_path: str = "dexter_bias.json", output_path: str = None) -> Dict[str, Any]:
    """
    Read existing dexter_bias.json and enrich with FinViz fundamental data.

    Adds to each ticker:
        - finviz_fundamentals: key metrics
        - finviz_risk: risk assessment
        - finviz_news: recent headlines

    Updates the decision if fundamentals suggest higher risk.
    """
    if output_path is None:
        output_path = bias_path

    bias_file = Path(bias_path)
    if not bias_file.exists():
        logger.warning(f"Bias file not found: {bias_path}")
        return {}

    try:
        bias_data = json.loads(bias_file.read_text())
    except Exception as e:
        logger.error(f"Failed to read bias file: {e}")
        return {}

    enriched = {}

    for ticker, existing_bias in bias_data.items():
        logger.info(f"Enriching {ticker} with FinViz data...")

        # Preserve existing Dexter analysis
        if isinstance(existing_bias, dict):
            enriched[ticker] = existing_bias.copy()
        else:
            enriched[ticker] = {'decision': existing_bias, 'reason': ''}

        # Fetch FinViz fundamentals
        fundamentals = get_stock_fundamentals(ticker)
        if fundamentals:
            # Assess risk
            risk_assessment = assess_fundamental_risk(fundamentals)

            # Add to enriched data
            enriched[ticker]['finviz_fundamentals'] = {
                'pe': fundamentals.get('pe'),
                'forward_pe': fundamentals.get('forward_pe'),
                'debt_equity': fundamentals.get('debt_equity'),
                'profit_margin': fundamentals.get('profit_margin'),
                'short_float': fundamentals.get('short_float'),
                'insider_trans': fundamentals.get('insider_trans'),
                'rsi_14': fundamentals.get('rsi_14'),
                'beta': fundamentals.get('beta'),
                'recommendation': fundamentals.get('recommendation'),
                'target_price': fundamentals.get('target_price'),
                'price': fundamentals.get('price'),
                'sma20': fundamentals.get('sma20'),
                'sma50': fundamentals.get('sma50'),
                'earnings_date': fundamentals.get('earnings_date'),
            }
            enriched[ticker]['finviz_risk'] = risk_assessment

            # Fetch recent news headlines
            news = get_stock_news(ticker, limit=3)
            if news:
                enriched[ticker]['finviz_news'] = news

            # Upgrade decision to 'avoid' if FinViz risk is high
            if risk_assessment['recommendation'] == 'avoid':
                if enriched[ticker].get('decision', '').lower() == 'ok':
                    enriched[ticker]['decision'] = 'caution'
                    original_reason = enriched[ticker].get('reason', '')
                    enriched[ticker]['reason'] = f"{original_reason} [FinViz: {', '.join(risk_assessment['risk_flags'])}]"

        # Rate limit between requests
        time.sleep(0.5)

    # Write enriched data
    try:
        Path(output_path).write_text(json.dumps(enriched, indent=2))
        logger.info(f"Enriched bias written to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write enriched bias: {e}")

    return enriched


def get_fundamentals_for_ml(tickers: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Fetch fundamental features for ML model training.

    Returns dict mapping ticker -> feature dict with normalized values.
    """
    features_by_ticker = {}

    for ticker in tickers:
        fundamentals = get_stock_fundamentals(ticker)
        if not fundamentals:
            continue

        # Extract ML-relevant features (numeric only)
        features_by_ticker[ticker] = {
            'PE_Ratio': fundamentals.get('pe') or 0,
            'Forward_PE': fundamentals.get('forward_pe') or 0,
            'PEG_Ratio': fundamentals.get('peg') or 0,
            'PS_Ratio': fundamentals.get('ps') or 0,
            'PB_Ratio': fundamentals.get('pb') or 0,
            'Debt_Equity': fundamentals.get('debt_equity') or 0,
            'ROE': fundamentals.get('roe') or 0,
            'ROI': fundamentals.get('roi') or 0,
            'Profit_Margin': fundamentals.get('profit_margin') or 0,
            'Gross_Margin': fundamentals.get('gross_margin') or 0,
            'EPS_Growth': fundamentals.get('eps_growth_yoy') or 0,
            'Sales_Growth': fundamentals.get('sales_growth_yoy') or 0,
            'Insider_Trans': fundamentals.get('insider_trans') or 0,
            'Inst_Trans': fundamentals.get('inst_trans') or 0,
            'Short_Float': fundamentals.get('short_float') or 0,
            'Short_Ratio': fundamentals.get('short_ratio') or 0,
            'Beta': fundamentals.get('beta') or 1,
            'Analyst_Recom': fundamentals.get('recommendation') or 3,  # 3 = Hold
            'SMA20_Dist': fundamentals.get('sma20') or 0,
            'SMA50_Dist': fundamentals.get('sma50') or 0,
            'SMA200_Dist': fundamentals.get('sma200') or 0,
            '52W_High_Dist': fundamentals.get('52w_high') or 0,
            '52W_Low_Dist': fundamentals.get('52w_low') or 0,
        }

        time.sleep(0.3)  # Rate limiting

    return features_by_ticker


# Helper functions for parsing FinViz string values

def _parse_float(value: str) -> Optional[float]:
    """Parse a string to float, handling '-' and empty values."""
    if not value or value == '-':
        return None
    try:
        # Remove commas and convert
        return float(value.replace(',', ''))
    except (ValueError, AttributeError):
        return None


def _parse_percent(value: str) -> Optional[float]:
    """Parse a percentage string like '5.23%' to float 5.23."""
    if not value or value == '-':
        return None
    try:
        return float(value.replace('%', '').replace(',', ''))
    except (ValueError, AttributeError):
        return None


def _parse_market_cap(value: str) -> Optional[float]:
    """Parse market cap like '2.5T' or '500M' to float in millions."""
    if not value or value == '-':
        return None
    try:
        value = value.upper().strip()
        multipliers = {'K': 0.001, 'M': 1, 'B': 1000, 'T': 1000000}
        for suffix, mult in multipliers.items():
            if value.endswith(suffix):
                return float(value[:-1]) * mult
        return float(value)
    except (ValueError, AttributeError):
        return None


def _parse_volume(value: str) -> Optional[float]:
    """Parse volume like '5.2M' to float."""
    if not value or value == '-':
        return None
    try:
        value = value.upper().strip()
        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
        for suffix, mult in multipliers.items():
            if value.endswith(suffix):
                return float(value[:-1]) * mult
        return float(value.replace(',', ''))
    except (ValueError, AttributeError):
        return None


# CLI for testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        print(f"\n=== FinViz Data for {ticker} ===\n")

        fundamentals = get_stock_fundamentals(ticker)
        if fundamentals:
            print("Fundamentals:")
            for k, v in fundamentals.items():
                if v is not None and v != '':
                    print(f"  {k}: {v}")

            print("\nRisk Assessment:")
            risk = assess_fundamental_risk(fundamentals)
            print(f"  Score: {risk['risk_score']}")
            print(f"  Flags: {risk['risk_flags']}")
            print(f"  Recommendation: {risk['recommendation']}")

            print("\nRecent News:")
            news = get_stock_news(ticker)
            for item in news:
                print(f"  - {item['title'][:60]}...")
        else:
            print("Failed to fetch data")
    else:
        print("Usage: python finviz_enrichment.py TICKER")
        print("       python finviz_enrichment.py --enrich  (to enrich dexter_bias.json)")

        if len(sys.argv) > 1 and sys.argv[1] == '--enrich':
            enrich_dexter_bias()
