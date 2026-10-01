"""Checks for the realized-net-equity gross daily ledger materializer."""

import pandas as pd
import pytest

from scripts.materialize_databento_gross_daily import build_daily_ledger


def _inputs():
    daily = pd.DataFrame(
        {
            "date": [
                "2019-01-02 00:00:00-05:00",
                "2019-01-03 00:00:00-05:00",
                "2019-01-04 00:00:00-05:00",
            ],
            "equity": [100090.0, 100090.0, 100080.0],
        }
    )
    trades = pd.DataFrame(
        {
            "entry_time": [
                "2019-01-02 10:00:00-05:00",
                "2019-01-02 11:00:00-05:00",
                "2019-01-04 10:00:00-05:00",
            ],
            "exit_time": [
                "2019-01-02 10:30:00-05:00",
                "2019-01-02 11:30:00-05:00",
                "2019-01-04 10:30:00-05:00",
            ],
            "gross_pnl": [80.0, 20.0, -8.0],
            "costs": [8.0, 2.0, 2.0],
            "net_pnl": [72.0, 18.0, -10.0],
        }
    )
    sessions = pd.Index(["2019-01-02", "2019-01-03", "2019-01-04"])
    return daily, trades, sessions


def test_uses_previous_realized_net_equity_and_keeps_no_trade_session():
    daily, trades, sessions = _inputs()
    ledger, checks = build_daily_ledger(daily, trades, sessions, 100000.0)

    assert ledger["date"].tolist() == sessions.tolist()
    assert ledger["start_equity"].tolist() == [100000.0, 100090.0, 100090.0]
    assert ledger["daily_gross_pnl"].tolist() == [100.0, 0.0, -8.0]
    assert ledger["daily_costs"].tolist() == [10.0, 0.0, 2.0]
    assert ledger["daily_net_pnl"].tolist() == [90.0, 0.0, -10.0]
    assert ledger["trade_count"].tolist() == [2, 0, 1]
    assert ledger["gross_return"].tolist() == pytest.approx([0.001, 0.0, -8 / 100090])
    assert ledger["net_return"].tolist() == pytest.approx([0.0009, 0.0, -10 / 100090])
    assert checks["trade_count"] == 3


def test_rejects_overnight_trade_before_materializing():
    daily, trades, sessions = _inputs()
    trades.loc[0, "exit_time"] = "2019-01-03 10:30:00-05:00"

    with pytest.raises(ValueError, match="BLOCKED_OVERNIGHT_POSITION_FOUND"):
        build_daily_ledger(daily, trades, sessions, 100000.0)


def test_rejects_naive_trade_timestamp():
    daily, trades, sessions = _inputs()
    trades.loc[0, "entry_time"] = "2019-01-02 10:00:00"

    with pytest.raises(ValueError, match="BLOCKED_TRADE_SESSION_MAPPING"):
        build_daily_ledger(daily, trades, sessions, 100000.0)
