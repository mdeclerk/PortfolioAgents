"""Stage 1: pure metric computations over snapshot data — no LLM, no I/O.

Every number the agents see is computed here; agents interpret, never calculate.
Series are date-indexed pandas frames, so cross-series values (IV vs HV) compare on
matching dates. NaN becomes None at the Pydantic boundary: a metric is None whenever
its input series is missing or too short.
"""

import math

import pandas as pd

from portfolio_agents.models import (
    DailyPoint,
    PortfolioMetrics,
    PortfolioSnapshot,
    PositionMetrics,
    PositionSnapshot,
)

TRADING_DAYS_1M = 21
TRADING_DAYS_3M = 63
TRADING_DAYS_1Y = 252


def compute_portfolio_metrics(snapshot: PortfolioSnapshot) -> PortfolioMetrics:
    values = {
        p.con_id: abs(p.market_value)
        for p in snapshot.positions
        if p.market_value is not None and p.market_value != 0
    }
    gross = sum(values.values())
    weights = {cid: v / gross for cid, v in values.items()} if gross > 0 else {}

    def exposure(key: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in snapshot.positions:
            if p.con_id in weights:
                name = getattr(p, key) or "unknown"
                out[name] = out.get(name, 0.0) + weights[p.con_id]
        return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))

    ranked = sorted(weights.values(), reverse=True)
    return PortfolioMetrics(
        gross_exposure=gross if gross > 0 else None,
        hhi=sum(w * w for w in ranked) if ranked else None,
        top3_concentration=sum(ranked[:3]) if ranked else None,
        currency_exposure=exposure("currency"),
        sector_exposure=exposure("sector"),
        asset_class_exposure=exposure("sec_type"),
        positions=[compute_position_metrics(p, weights.get(p.con_id)) for p in snapshot.positions],
    )


def compute_position_metrics(position: PositionSnapshot, weight: float | None) -> PositionMetrics:
    bars = _frame(position.bars)
    close = bars["close"]
    volume = bars["volume"]
    iv = _closes(position.iv_series)
    hv = _closes(position.hv_series)

    last_close = _last(close)
    sma_50 = _last(close.rolling(50).mean())
    sma_200 = _last(close.rolling(200).mean())

    sentiment = position.sentiment
    put_call = None
    if sentiment and sentiment.put_volume is not None and (sentiment.call_volume or 0) > 0:
        put_call = sentiment.put_volume / sentiment.call_volume

    return PositionMetrics(
        symbol=position.symbol,
        con_id=position.con_id,
        weight=weight,
        as_of=close.index[-1].date() if not close.empty else None,
        last_close=last_close,
        return_1m=_last(close.pct_change(TRADING_DAYS_1M)),
        return_3m=_last(close.pct_change(TRADING_DAYS_3M)),
        return_1y=_last(close.pct_change(len(close) - 1)) if len(close) > 1 else None,
        max_drawdown_1y=_f((close / close.cummax() - 1).min()) if not close.empty else None,
        range_position_52w=_range_position(close),
        rsi_14=_rsi(close),
        sma_50=sma_50,
        sma_200=sma_200,
        price_vs_sma50=_vs(last_close, sma_50),
        price_vs_sma200=_vs(last_close, sma_200),
        sma_cross=_cross(sma_50, sma_200),
        atr_14=_atr(bars),
        sharpe_1y=_sharpe(close),
        iv=_last(iv),
        hv=_last(hv),
        iv_hv_ratio=_iv_hv_ratio(iv, hv),
        iv_rank_1y=_range_position(iv),
        put_call_ratio=put_call,
        volume_vs_50d=_volume_vs_avg(volume, 50),
        up_down_volume_ratio=_up_down_volume_ratio(close, volume),
    )


_COLUMNS = ["open", "high", "low", "close", "volume"]


def _frame(points: list[DailyPoint] | None) -> pd.DataFrame:
    frame = pd.DataFrame(
        [point.model_dump() for point in points or []], columns=["date", *_COLUMNS]
    )
    frame.index = pd.DatetimeIndex(frame.pop("date"))
    frame.index.name = None
    return frame.sort_index()


def _closes(points: list[DailyPoint] | None) -> pd.Series:
    return _frame(points)["close"]


def _f(value: float | None) -> float | None:
    """NaN → None at the Pydantic boundary."""
    return float(value) if pd.notna(value) else None


def _last(series: pd.Series) -> float | None:
    return _f(series.iloc[-1]) if not series.empty else None


def _vs(price: float | None, sma: float | None) -> float | None:
    return price / sma - 1 if price is not None and sma else None


def _cross(sma_50: float | None, sma_200: float | None) -> str | None:
    if sma_50 is None or sma_200 is None or sma_50 == sma_200:
        return None
    return "golden" if sma_50 > sma_200 else "death"


def _range_position(series: pd.Series) -> float | None:
    """Where the last value sits in the window's range: 0 = at low, 1 = at high."""
    if series.empty:
        return None
    low, high = series.min(), series.max()
    return _f((series.iloc[-1] - low) / (high - low)) if high > low else None


def _rsi(close: pd.Series, n: int = 14) -> float | None:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return _last(100 * gain / (gain + loss))  # flat series → 0/0 → NaN → None


def _atr(bars: pd.DataFrame, n: int = 14) -> float | None:
    if len(bars) <= n:
        return None
    prev_close = bars["close"].shift()
    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _last(true_range.ewm(alpha=1 / n, adjust=False, min_periods=n).mean())


def _sharpe(close: pd.Series, min_observations: int = 20) -> float | None:
    """Annualized Sharpe over daily returns, rf = 0."""
    returns = close.pct_change().dropna()
    if len(returns) < min_observations:
        return None
    spread = returns.std()
    if spread == 0:
        return None
    return _f(returns.mean() / spread * math.sqrt(TRADING_DAYS_1Y))


def _iv_hv_ratio(iv: pd.Series, hv: pd.Series) -> float | None:
    """IV over HV on the most recent date where both series have a value."""
    both = pd.DataFrame({"iv": iv, "hv": hv}).dropna()
    if both.empty:
        return None
    last = both.iloc[-1]
    return _f(last["iv"] / last["hv"]) if last["hv"] > 0 else None


def _volume_vs_avg(volume: pd.Series, n: int) -> float | None:
    volumes = volume[volume >= 0]
    if len(volumes) < n:
        return None
    avg = volumes.tail(n).mean()
    return _f(volumes.iloc[-1] / avg) if avg > 0 else None


def _up_down_volume_ratio(close: pd.Series, volume: pd.Series) -> float | None:
    """Average volume on up days over average volume on down days."""
    delta = close.diff()
    valid = volume >= 0
    up = volume[valid & (delta > 0)]
    down = volume[valid & (delta < 0)]
    if up.empty or down.empty or down.mean() == 0:
        return None
    return _f(up.mean() / down.mean())
