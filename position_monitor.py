"""
Position Monitor - Enforces stop losses on existing positions

This runs as part of each trading cycle to check if any positions
have breached their stop loss or take profit levels, and closes them.

This is a safety net in case bracket orders fail or weren't placed.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)


def check_and_enforce_stops(api, config: Dict) -> List[Dict]:
    """
    Check all open positions and close any that have breached stop loss or take profit.

    Args:
        api: Alpaca REST API client
        config: Configuration dict with stop_loss_pct, take_profit_pct

    Returns:
        List of actions taken (for logging)
    """
    actions = []

    stop_loss_pct = config.get('stop_loss_pct', 0.05)
    take_profit_pct = config.get('take_profit_pct', 0.10)

    # Optional: trailing stop settings
    trailing_stop = config.get('trailing_stop', {})
    use_trailing = trailing_stop.get('enabled', False)
    trail_pct = trailing_stop.get('trail_pct', 0.03)  # 3% trailing stop

    try:
        positions = api.list_positions()
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return actions

    if not positions:
        logger.debug("No open positions to monitor")
        return actions

    logger.info(f"Monitoring {len(positions)} open positions for stop loss enforcement...")

    for pos in positions:
        symbol = pos.symbol
        qty = float(pos.qty)
        entry_price = float(pos.avg_entry_price)
        current_price = float(pos.current_price)
        unrealized_pl_pct = float(pos.unrealized_plpc)

        # Calculate stop and target prices
        stop_price = entry_price * (1 - stop_loss_pct)
        target_price = entry_price * (1 + take_profit_pct)

        action = None
        reason = None

        # Check stop loss
        if current_price <= stop_price:
            action = 'STOP_LOSS'
            reason = f"Price ${current_price:.2f} <= Stop ${stop_price:.2f} ({unrealized_pl_pct*100:+.1f}%)"

        # Check take profit
        elif current_price >= target_price:
            action = 'TAKE_PROFIT'
            reason = f"Price ${current_price:.2f} >= Target ${target_price:.2f} ({unrealized_pl_pct*100:+.1f}%)"

        # Check trailing stop (if enabled and position is profitable)
        elif use_trailing and unrealized_pl_pct > trail_pct:
            # Calculate trailing stop from highest point
            # For simplicity, use current profit - trail_pct as the trigger
            trail_trigger = entry_price * (1 + unrealized_pl_pct - trail_pct)
            if current_price <= trail_trigger:
                action = 'TRAILING_STOP'
                reason = f"Trailing stop triggered at ${current_price:.2f}"

        if action:
            logger.warning(f"🚨 {symbol}: {action} - {reason}")

            try:
                # Close the position
                order = api.submit_order(
                    symbol=symbol,
                    qty=abs(int(qty)) if not is_crypto(symbol) else abs(qty),
                    side='sell',
                    type='market',
                    time_in_force='day' if not is_crypto(symbol) else 'gtc'
                )

                actions.append({
                    'symbol': symbol,
                    'action': action,
                    'reason': reason,
                    'qty': qty,
                    'price': current_price,
                    'order_id': order.id,
                    'status': 'submitted'
                })

                logger.info(f"✅ {symbol}: {action} order submitted - {order.id}")

            except Exception as e:
                logger.error(f"❌ Failed to close {symbol}: {e}")
                actions.append({
                    'symbol': symbol,
                    'action': action,
                    'reason': reason,
                    'status': 'failed',
                    'error': str(e)
                })
        else:
            # Position is within normal range
            logger.debug(f"{symbol}: OK - Current ${current_price:.2f}, "
                        f"Stop ${stop_price:.2f}, Target ${target_price:.2f}, "
                        f"P&L: {unrealized_pl_pct*100:+.1f}%")

    if actions:
        logger.info(f"Position monitor took {len(actions)} actions")

    return actions


def is_crypto(symbol: str) -> bool:
    """Check if symbol is a crypto pair."""
    return '/' in symbol or symbol.endswith('USD') and symbol not in ['USD']


def get_position_summary(api) -> str:
    """Get a formatted summary of all positions."""
    try:
        positions = api.list_positions()
    except Exception as e:
        return f"Error fetching positions: {e}"

    if not positions:
        return "No open positions"

    lines = ["POSITION SUMMARY", "=" * 50]
    total_value = 0
    total_pnl = 0

    for pos in positions:
        pnl = float(pos.unrealized_pl)
        pnl_pct = float(pos.unrealized_plpc) * 100
        value = float(pos.market_value)
        emoji = "🟢" if pnl >= 0 else "🔴"

        lines.append(f"{emoji} {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f}")
        lines.append(f"   Current: ${float(pos.current_price):.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)")

        total_value += value
        total_pnl += pnl

    lines.append("=" * 50)
    emoji = "🟢" if total_pnl >= 0 else "🔴"
    lines.append(f"{emoji} Total: ${total_value:,.2f} | P&L: ${total_pnl:+,.2f}")

    return "\n".join(lines)
