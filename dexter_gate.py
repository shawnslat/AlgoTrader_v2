import json
from pathlib import Path
from typing import Dict, Any


class DexterGate:
    """
    Simple hook to allow/deny trades based on a bias file produced by Dexter.
    If dexter_bias.json exists and marks a ticker as 'avoid', we block the trade.
    """

    def __init__(self, bias_path: str = "dexter_bias.json"):
        self.bias_path = Path(bias_path)
        self.bias = self._load_bias()

    def _load_bias(self) -> Dict[str, Any]:
        if self.bias_path.exists():
            try:
                return json.loads(self.bias_path.read_text())
            except Exception:
                return {}
        return {}

    def refresh(self):
        self.bias = self._load_bias()

    def get_bias(self, ticker: str) -> Dict[str, Any]:
        """
        Get the full bias data for a ticker (for confidence scoring).
        Returns dict with 'bias', 'reasoning', 'fundamentals', etc.
        """
        bias_data = self.bias.get(ticker)
        if bias_data is None:
            return {}
        if isinstance(bias_data, str):
            return {'bias': bias_data}
        if isinstance(bias_data, dict):
            return bias_data
        return {}

    def should_allow(self, ticker: str, trades_remaining: int, context: Dict[str, Any]) -> bool:
        """
        Basic gating: if no trades remaining, block; if Dexter says 'avoid', block.
        """
        if trades_remaining <= 0:
            return False
        bias = self.bias.get(ticker) or {}
        if isinstance(bias, str) and bias.lower().strip() == "avoid":
            return False
        if isinstance(bias, dict):
            if bias.get("decision", "").lower() == "avoid":
                return False
            if bias.get("allow") is False:
                return False
        return True
