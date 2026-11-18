"""Market data providers."""
from __future__ import annotations

from ..config import Config, alpaca_keys
from .base import DataError, DataProvider, Quote


def resolve_provider_name(cfg: Config) -> str:
    name = str(cfg.data.provider).lower()
    if name == "auto":
        return "alpaca" if alpaca_keys() else "yahoo"
    if name not in ("alpaca", "yahoo"):
        raise DataError("data.provider must be auto, alpaca or yahoo (got %r)" % cfg.data.provider)
    return name


def get_provider(cfg: Config) -> DataProvider:
    name = resolve_provider_name(cfg)
    if name == "alpaca":
        from .alpaca import AlpacaProvider

        return AlpacaProvider(cfg)
    from .yahoo import YahooProvider

    return YahooProvider(cfg)


__all__ = ["DataError", "DataProvider", "Quote", "get_provider", "resolve_provider_name"]
