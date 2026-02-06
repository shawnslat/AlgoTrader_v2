from datetime import datetime, timedelta
from collections import defaultdict


class DayTradeGuard:
    """
    Lightweight PDT/day-trade limiter.
    Tracks intraday round trips and blocks new entries once the limit is hit.
    """

    def __init__(self, max_day_trades: int = 2):
        self.max_day_trades = max_day_trades
        self.day_trade_counts = defaultdict(int)  # date string -> count
        self.open_entries = {}  # ticker -> entry date string

    @staticmethod
    def _date_str(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d")

    def _prune_old(self, now: datetime):
        cutoff = now - timedelta(days=5)
        # Normalize stored dates to naive for comparison with naive cutoff
        cutoff_naive = cutoff.replace(tzinfo=None)
        to_delete = []
        for d in self.day_trade_counts:
            try:
                dt_val = datetime.fromisoformat(d)
            except Exception:
                continue
            dt_val_naive = dt_val.replace(tzinfo=None)
            if dt_val_naive < cutoff_naive:
                to_delete.append(d)
        for d in to_delete:
            del self.day_trade_counts[d]

    def remaining_today(self, now: datetime) -> int:
        self._prune_old(now)
        today = self._date_str(now.replace(tzinfo=None))
        used = self.day_trade_counts.get(today, 0)
        return max(self.max_day_trades - used, 0)

    def can_open(self, now: datetime) -> bool:
        return self.remaining_today(now) > 0

    def record_entry(self, ticker: str, now: datetime):
        self.open_entries[ticker] = self._date_str(now.replace(tzinfo=None))

    def record_exit(self, ticker: str, now: datetime):
        entry_date = self.open_entries.get(ticker)
        if entry_date:
            exit_date = self._date_str(now.replace(tzinfo=None))
            if entry_date == exit_date:
                self.day_trade_counts[exit_date] += 1
            del self.open_entries[ticker]

    def current_open_entry_date(self, ticker: str) -> str:
        return self.open_entries.get(ticker, "")
