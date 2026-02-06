"""
Signal Confidence Scoring System

Calculates trade confidence based on agreement between multiple signal sources:
- Dexter bias (AI research)
- News sentiment
- Technical indicators
- FinViz fundamentals

When signals agree → High confidence → Full position size
When signals conflict → Low confidence → Reduced position or skip
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1


@dataclass
class SignalSource:
    """Individual signal from a data source."""
    name: str
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    reason: str


@dataclass
class ConfidenceScore:
    """Overall confidence score for a trading decision."""
    ticker: str
    score: float  # 0.0 to 1.0
    direction: SignalDirection
    position_multiplier: float  # How much to scale position (0.0 to 1.0)
    signals: list
    agreement_pct: float
    recommendation: str  # "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "SKIP"


def calculate_confidence(
    ticker: str,
    dexter_bias: Optional[Dict] = None,
    sentiment_score: Optional[float] = None,  # -1.0 to 1.0
    technical_signal: Optional[str] = None,  # "BUY", "SELL", "HOLD"
    ml_prediction: Optional[int] = None,  # 1 = up, 0 = down
    finviz_risk: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> ConfidenceScore:
    """
    Calculate confidence score based on signal agreement.

    Args:
        ticker: Stock ticker symbol
        dexter_bias: Dexter bias dict with 'bias' key ("bullish", "bearish", "neutral")
        sentiment_score: Aggregated sentiment (-1.0 bearish to 1.0 bullish)
        technical_signal: Technical indicator signal
        ml_prediction: ML model prediction (1=up, 0=down)
        finviz_risk: FinViz risk assessment with 'risk_score' and 'flags'
        config: Configuration dict with confidence settings

    Returns:
        ConfidenceScore with overall assessment
    """
    config = config or {}
    conf_cfg = config.get('confidence', {})

    # Weights for each signal source (configurable)
    weights = {
        'dexter': conf_cfg.get('weight_dexter', 0.30),
        'sentiment': conf_cfg.get('weight_sentiment', 0.20),
        'technical': conf_cfg.get('weight_technical', 0.20),
        'ml': conf_cfg.get('weight_ml', 0.20),
        'fundamental': conf_cfg.get('weight_fundamental', 0.10),
    }

    signals = []

    # 1. Dexter Bias Signal
    if dexter_bias:
        bias = dexter_bias.get('bias', '').lower()
        if 'bullish' in bias:
            direction = SignalDirection.BULLISH
            strength = 0.8 if 'strong' in bias else 0.6
        elif 'bearish' in bias:
            direction = SignalDirection.BEARISH
            strength = 0.8 if 'strong' in bias else 0.6
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.3

        signals.append(SignalSource(
            name='dexter',
            direction=direction,
            strength=strength,
            reason=dexter_bias.get('reasoning', bias)[:100]
        ))

    # 2. Sentiment Signal
    if sentiment_score is not None:
        if sentiment_score > 0.2:
            direction = SignalDirection.BULLISH
            strength = min(1.0, abs(sentiment_score))
        elif sentiment_score < -0.2:
            direction = SignalDirection.BEARISH
            strength = min(1.0, abs(sentiment_score))
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.3

        signals.append(SignalSource(
            name='sentiment',
            direction=direction,
            strength=strength,
            reason=f"Sentiment score: {sentiment_score:.2f}"
        ))

    # 3. Technical Signal
    if technical_signal:
        sig_upper = technical_signal.upper()
        if sig_upper == 'BUY':
            direction = SignalDirection.BULLISH
            strength = 0.7
        elif sig_upper == 'SELL':
            direction = SignalDirection.BEARISH
            strength = 0.7
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.3

        signals.append(SignalSource(
            name='technical',
            direction=direction,
            strength=strength,
            reason=f"Technical: {technical_signal}"
        ))

    # 4. ML Prediction Signal
    if ml_prediction is not None:
        if ml_prediction == 1:
            direction = SignalDirection.BULLISH
            strength = 0.6  # ML gets moderate weight
        else:
            direction = SignalDirection.BEARISH
            strength = 0.6

        signals.append(SignalSource(
            name='ml',
            direction=direction,
            strength=strength,
            reason=f"ML predicts: {'UP' if ml_prediction == 1 else 'DOWN'}"
        ))

    # 5. Fundamental/Risk Signal (from FinViz)
    if finviz_risk:
        risk_score = finviz_risk.get('risk_score', 0)
        flags = finviz_risk.get('flags', [])

        # High risk = bearish signal, low risk = neutral (not bullish by itself)
        if risk_score >= 3:  # High risk
            direction = SignalDirection.BEARISH
            strength = min(1.0, risk_score / 5)
        elif risk_score <= 1:  # Low risk
            direction = SignalDirection.NEUTRAL  # Low risk doesn't mean buy
            strength = 0.2
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.3

        signals.append(SignalSource(
            name='fundamental',
            direction=direction,
            strength=strength,
            reason=f"Risk: {risk_score}, Flags: {', '.join(flags[:3])}" if flags else f"Risk score: {risk_score}"
        ))

    # Calculate agreement and confidence
    if not signals:
        return ConfidenceScore(
            ticker=ticker,
            score=0.0,
            direction=SignalDirection.NEUTRAL,
            position_multiplier=0.0,
            signals=[],
            agreement_pct=0.0,
            recommendation="SKIP"
        )

    # Count directions
    bullish_count = sum(1 for s in signals if s.direction == SignalDirection.BULLISH)
    bearish_count = sum(1 for s in signals if s.direction == SignalDirection.BEARISH)
    neutral_count = sum(1 for s in signals if s.direction == SignalDirection.NEUTRAL)

    total_signals = len(signals)

    # Determine dominant direction
    if bullish_count > bearish_count and bullish_count > neutral_count:
        dominant_direction = SignalDirection.BULLISH
        agreement_count = bullish_count
    elif bearish_count > bullish_count and bearish_count > neutral_count:
        dominant_direction = SignalDirection.BEARISH
        agreement_count = bearish_count
    else:
        dominant_direction = SignalDirection.NEUTRAL
        agreement_count = neutral_count

    agreement_pct = agreement_count / total_signals

    # Calculate weighted confidence score
    weighted_score = 0.0
    total_weight = 0.0

    for signal in signals:
        weight = weights.get(signal.name, 0.1)
        # Signals agreeing with dominant direction add to score
        if signal.direction == dominant_direction:
            weighted_score += weight * signal.strength
        elif signal.direction == SignalDirection.NEUTRAL:
            weighted_score += weight * 0.3  # Neutral gives partial credit
        else:
            weighted_score -= weight * signal.strength * 0.5  # Disagreement reduces score
        total_weight += weight

    # Normalize score to 0-1 range
    if total_weight > 0:
        confidence_score = max(0.0, min(1.0, (weighted_score / total_weight + 0.5)))
    else:
        confidence_score = 0.5

    # Determine position multiplier based on confidence
    min_confidence = conf_cfg.get('min_confidence_to_trade', 0.4)
    full_confidence = conf_cfg.get('full_confidence_threshold', 0.7)

    if confidence_score < min_confidence:
        position_multiplier = 0.0  # Don't trade
    elif confidence_score >= full_confidence:
        position_multiplier = 1.0  # Full position
    else:
        # Linear scale between min and full
        position_multiplier = (confidence_score - min_confidence) / (full_confidence - min_confidence)

    # Generate recommendation
    if position_multiplier == 0.0:
        recommendation = "SKIP"
    elif dominant_direction == SignalDirection.BULLISH:
        if confidence_score >= 0.7:
            recommendation = "STRONG_BUY"
        else:
            recommendation = "BUY"
    elif dominant_direction == SignalDirection.BEARISH:
        if confidence_score >= 0.7:
            recommendation = "STRONG_SELL"
        else:
            recommendation = "SELL"
    else:
        recommendation = "HOLD"

    result = ConfidenceScore(
        ticker=ticker,
        score=confidence_score,
        direction=dominant_direction,
        position_multiplier=position_multiplier,
        signals=signals,
        agreement_pct=agreement_pct,
        recommendation=recommendation
    )

    logger.info(
        f"[{ticker}] Confidence: {confidence_score:.2f}, "
        f"Agreement: {agreement_pct:.0%}, "
        f"Direction: {dominant_direction.name}, "
        f"Position: {position_multiplier:.0%}, "
        f"Recommendation: {recommendation}"
    )

    return result


def get_aggregated_sentiment(sentiment_results: list, ticker: str) -> Optional[float]:
    """
    Aggregate sentiment results for a specific ticker.

    Args:
        sentiment_results: List of sentiment dicts with 'ticker', 'label', 'magnitude', 'recency_weight'
        ticker: Ticker to filter for

    Returns:
        Aggregated sentiment score from -1.0 (bearish) to 1.0 (bullish)
    """
    ticker_sentiments = [s for s in sentiment_results if s.get('ticker') == ticker]

    if not ticker_sentiments:
        return None

    weighted_sum = 0.0
    total_weight = 0.0

    for s in ticker_sentiments:
        label = s.get('label', 'Neutral')
        magnitude = s.get('magnitude', 1.0)
        recency = s.get('recency_weight', 0.5)

        # Convert label to numeric
        if label == 'Positive':
            value = 1.0
        elif label == 'Negative':
            value = -1.0
        else:
            value = 0.0

        # Weight by magnitude and recency
        weight = magnitude * recency
        weighted_sum += value * weight
        total_weight += weight

    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.0


def format_confidence_report(scores: list) -> str:
    """
    Format confidence scores into a readable report.

    Args:
        scores: List of ConfidenceScore objects

    Returns:
        Formatted string report
    """
    if not scores:
        return "No confidence scores calculated."

    lines = ["=" * 60, "SIGNAL CONFIDENCE REPORT", "=" * 60, ""]

    # Sort by confidence score descending
    sorted_scores = sorted(scores, key=lambda x: x.score, reverse=True)

    for cs in sorted_scores:
        direction_emoji = "🟢" if cs.direction == SignalDirection.BULLISH else "🔴" if cs.direction == SignalDirection.BEARISH else "⚪"

        lines.append(f"{direction_emoji} {cs.ticker}: {cs.recommendation}")
        lines.append(f"   Confidence: {cs.score:.0%} | Agreement: {cs.agreement_pct:.0%} | Position: {cs.position_multiplier:.0%}")

        # Show individual signals
        for sig in cs.signals:
            sig_icon = "↑" if sig.direction == SignalDirection.BULLISH else "↓" if sig.direction == SignalDirection.BEARISH else "→"
            lines.append(f"   {sig_icon} {sig.name}: {sig.reason[:50]}")

        lines.append("")

    # Summary
    buy_count = sum(1 for cs in scores if cs.recommendation in ["BUY", "STRONG_BUY"])
    sell_count = sum(1 for cs in scores if cs.recommendation in ["SELL", "STRONG_SELL"])
    skip_count = sum(1 for cs in scores if cs.recommendation in ["SKIP", "HOLD"])

    lines.append("-" * 60)
    lines.append(f"Summary: {buy_count} BUY | {sell_count} SELL | {skip_count} SKIP/HOLD")
    lines.append("=" * 60)

    return "\n".join(lines)
