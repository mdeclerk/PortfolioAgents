"""compute_position_metrics / compute_portfolio_metrics: hand-checkable numbers, and the
degradation contract — a metric is None whenever its input is missing, too short, or flat."""

import datetime as dt

import pytest

from factories import make_account, make_bars, make_position
from portfolio_agents.metrics import compute_portfolio_metrics, compute_position_metrics
from portfolio_agents.models import PortfolioSnapshot, SentimentTicks

TAKEN_AT = dt.datetime(2025, 6, 1, 12, 0, tzinfo=dt.UTC)


def test_rising_ramp_trend_metrics():
    closes = [100.0 + i for i in range(260)]
    position = make_position(bars=make_bars(closes))
    m = compute_position_metrics(position, weight=0.5)
    assert m.weight == 0.5
    assert m.as_of == position.bars[-1].date
    assert m.last_close == closes[-1]
    assert m.return_1m == pytest.approx(closes[-1] / closes[-22] - 1)
    assert m.return_3m == pytest.approx(closes[-1] / closes[-64] - 1)
    assert m.return_1y == pytest.approx(closes[-1] / closes[0] - 1)
    assert m.max_drawdown_1y == 0.0
    assert m.range_position_52w == 1.0
    assert m.rsi_14 == 100.0  # gains only
    assert m.sma_cross == "golden"
    assert m.price_vs_sma50 > 0
    assert m.price_vs_sma200 > 0
    assert m.atr_14 == pytest.approx(1.0)  # |close - prev_close| == 1 every day
    assert m.sharpe_1y > 0


def test_declining_series_drawdown_and_death_cross():
    closes = [300.0 - i for i in range(260)]
    m = compute_position_metrics(make_position(bars=make_bars(closes)), weight=None)
    assert m.max_drawdown_1y == pytest.approx(closes[-1] / closes[0] - 1)
    assert m.range_position_52w == 0.0
    assert m.rsi_14 == 0.0  # losses only
    assert m.sma_cross == "death"
    assert m.sharpe_1y < 0


def test_missing_series_yields_all_none():
    m = compute_position_metrics(make_position(), weight=None)
    values = m.model_dump(exclude={"symbol", "con_id"})
    assert all(value is None for value in values.values())


def test_short_series_degrades_windowed_metrics():
    closes = [100.0, 102.0, 101.0, 103.0, 104.0, 102.0, 105.0, 106.0, 104.0, 107.0]
    m = compute_position_metrics(make_position(bars=make_bars(closes)), weight=None)
    assert m.last_close == 107.0
    assert m.return_1y == pytest.approx(0.07)
    unmet_windows = ("return_1m", "rsi_14", "sma_50", "sma_200", "atr_14", "sharpe_1y")
    for field in unmet_windows:
        assert getattr(m, field) is None, field
    assert m.volume_vs_50d is None


def test_flat_series_yields_no_signal():
    m = compute_position_metrics(make_position(bars=make_bars([50.0] * 60)), weight=None)
    assert m.return_1m == 0.0
    assert m.rsi_14 is None  # gain and loss both zero
    assert m.sharpe_1y is None  # zero variance
    assert m.range_position_52w is None  # high == low
    assert m.up_down_volume_ratio is None  # no up or down days


def test_iv_hv_ratio_aligns_on_last_common_date():
    iv = make_bars([0.30, 0.32, 0.40], volume=-1.0)
    hv = make_bars([0.20, 0.16], volume=-1.0)
    m = compute_position_metrics(make_position(iv_series=iv, hv_series=hv), weight=None)
    assert m.iv == pytest.approx(0.40)
    assert m.hv == pytest.approx(0.16)
    assert m.iv_hv_ratio == pytest.approx(0.32 / 0.16)  # the last date both series share
    assert m.iv_rank_1y == 1.0  # last IV value is the window high


def test_iv_hv_ratio_none_when_hv_is_zero():
    position = make_position(
        iv_series=make_bars([0.3], volume=-1.0), hv_series=make_bars([0.0], volume=-1.0)
    )
    assert compute_position_metrics(position, weight=None).iv_hv_ratio is None


def test_put_call_ratio_guards():
    def ratio(call: float | None, put: float | None) -> float | None:
        ticks = SentimentTicks(
            call_volume=call, put_volume=put, shortable_shares=None, shortable=None
        )
        return compute_position_metrics(make_position(sentiment=ticks), weight=None).put_call_ratio

    assert ratio(200.0, 150.0) == pytest.approx(0.75)
    assert ratio(0.0, 150.0) is None  # zero call volume
    assert ratio(200.0, None) is None
    assert compute_position_metrics(make_position(), weight=None).put_call_ratio is None


def test_volume_vs_50d_average():
    closes = [100.0 + i for i in range(60)]
    volumes = [1000.0] * 59 + [2000.0]
    m = compute_position_metrics(make_position(bars=make_bars(closes, volumes=volumes)), None)
    average = (49 * 1000.0 + 2000.0) / 50
    assert m.volume_vs_50d == pytest.approx(2000.0 / average)
    assert m.up_down_volume_ratio is None  # a ramp has no down days


def test_up_down_volume_ratio():
    closes = [100.0, 101.0, 100.0, 102.0, 101.0]
    volumes = [1000.0, 3000.0, 1000.0, 3000.0, 2000.0]
    m = compute_position_metrics(make_position(bars=make_bars(closes, volumes=volumes)), None)
    assert m.up_down_volume_ratio == pytest.approx(3000.0 / 1500.0)


def test_placeholder_volume_is_ignored():
    closes = [100.0 + (i % 3) for i in range(60)]  # mixed up/down days
    m = compute_position_metrics(make_position(bars=make_bars(closes, volume=-1.0)), None)
    assert m.volume_vs_50d is None
    assert m.up_down_volume_ratio is None


def test_portfolio_weights_concentration_and_exposures():
    snapshot = PortfolioSnapshot(
        taken_at=TAKEN_AT,
        account=make_account(),
        positions=[
            make_position(con_id=1, symbol="AAPL", market_value=500.0, sector="Technology"),
            make_position(
                con_id=2, symbol="NESN", market_value=300.0, currency="CHF", sector="Food"
            ),
            make_position(con_id=3, symbol="SHRT", market_value=-200.0),
            make_position(con_id=4, symbol="NOVAL", market_value=None),
        ],
    )
    metrics = compute_portfolio_metrics(snapshot)
    assert metrics.gross_exposure == pytest.approx(1000.0)  # shorts count at |value|
    assert metrics.hhi == pytest.approx(0.5**2 + 0.3**2 + 0.2**2)
    assert metrics.top3_concentration == pytest.approx(1.0)
    assert metrics.currency_exposure == pytest.approx({"USD": 0.7, "CHF": 0.3})
    expected_sectors = {"Technology": 0.5, "Food": 0.3, "unknown": 0.2}
    assert metrics.sector_exposure == pytest.approx(expected_sectors)
    assert list(metrics.sector_exposure) == ["Technology", "Food", "unknown"]  # sorted descending
    weights = {m.symbol: m.weight for m in metrics.positions}
    assert weights["AAPL"] == pytest.approx(0.5)
    assert weights["NOVAL"] is None  # no market value → not in the weight basis


def test_empty_portfolio_degrades_to_none():
    snapshot = PortfolioSnapshot(taken_at=TAKEN_AT, account=make_account(), positions=[])
    metrics = compute_portfolio_metrics(snapshot)
    assert metrics.gross_exposure is None
    assert metrics.hhi is None
    assert metrics.top3_concentration is None
    assert metrics.currency_exposure == {}
    assert metrics.positions == []
