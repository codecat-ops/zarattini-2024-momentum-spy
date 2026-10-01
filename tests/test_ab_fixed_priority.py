"""Fixed-priority A/B timing tests using hand-checked execution intervals."""

import pandas as pd
import pytest

from scripts.validate_ab_fixed_priority import combine_daily, control_comparison, strategy_paths


def _ledgers():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    a = pd.DataFrame(
        {
            "date": dates,
            "gross_return": [0.01, 0.02, 0.03, 0.04],
            "daily_gross_pnl": [100, 200, 300, 400],
            "trade_count": [1, 2, 3, 4],
        }
    )
    b = pd.DataFrame(
        {
            "date": dates,
            "exec_price": ["100", "110", "104.5", "106.59"],
            "cash_distribution": ["0", "0", "0", "0"],
            "position": ["1", "1", "0", "0"],
            "signal_action": ["ENTRY", "HOLD", "EXIT", "FLAT"],
            "execution_interval_total_return": ["", "0.1", "-0.05", "0.02"],
            "strategy_return": ["", "0.1", "-0.05", "0"],
        }
    )
    return a, b


def test_entry_keeps_a_then_hold_and_exit_use_b_then_a_resumes():
    a, b = _ledgers()
    combined = combine_daily(a, b)

    assert combined["selected_source"].tolist() == ["A", "B", "B", "A"]
    assert combined["combined_return"].tolist() == pytest.approx([0.01, 0.1, -0.05, 0.04])
    assert combined["b_position_before_execution"].tolist() == [0, 1, 1, 0]
    assert combined["suppressed_a_trade_count"].tolist() == [0, 2, 3, 0]
    assert combined["combined_end_equity"].iloc[-1] == pytest.approx(
        1.01 * 1.1 * 0.95 * 1.04
    )


def test_rejects_return_recorded_against_current_position():
    a, b = _ledgers()
    b.loc[1, "strategy_return"] = "0"  # Prior day's ENTRY must earn this interval.

    with pytest.raises(ValueError, match="BLOCKED_B_RETURN_ALIGNMENT"):
        combine_daily(a, b)


def test_rejects_misaligned_session_dates():
    a, b = _ledgers()
    b.loc[2, "date"] = "2020-01-07"

    with pytest.raises(ValueError, match="SESSION_DATE_MISMATCH"):
        combine_daily(a, b)


def test_control_uses_separate_ledger_and_outputs():
    daily, output_daily, output_summary, expected_hash = strategy_paths("RSI3_ONLY")
    assert daily.name == "RSI3_ONLY_daily.csv"
    assert output_daily != "ab_fixed_priority_combined_daily.csv"
    assert output_summary != "ab_fixed_priority_validation.json"
    assert expected_hash == "f1e4c90460194d28bdf2942f6c3217adfc6a98fc21526c5ac39638f70f8833ab"


def test_control_comparison_prefers_ibs_when_risk_metrics_worsen():
    ibs = {"cagr": 0.27, "sharpe_rf_0": 1.67, "max_drawdown": -0.11, "calmar": 2.3}
    only = {"cagr": 0.29, "sharpe_rf_0": 1.59, "max_drawdown": -0.19, "calmar": 1.47}
    comparison = control_comparison(only, ibs, {"b_priority_sessions": 318,
        "suppressed_a_gross_pnl_baseline_usd": 10911}, {"b_priority_sessions": 176,
        "suppressed_a_gross_pnl_baseline_usd": 5669})
    assert comparison["verdict"] == "RSI3_IBS_COMBINATION_PREFERRED"
    assert comparison["delta"]["cagr"] == pytest.approx(0.02)
    assert comparison["delta"]["max_drawdown"] == pytest.approx(-0.08)
    assert comparison["delta"]["b_priority_sessions"] == 142
