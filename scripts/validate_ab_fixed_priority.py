"""Validate fixed-priority A/B returns from frozen daily ledgers only.

B's post-execution position on session t controls the execution-to-execution
return ending on session t+1. No strategy evaluator is called here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
B_ROOT = ROOT.parent / "spy-mean-reversion"
A_DAILY = ROOT / "reports/databento_2019_2025_baseline_daily_gross.csv"
B_DAILY = B_ROOT / "reports/causal_preclose/RSI3_IBS_daily.csv"
B_SUMMARY = B_ROOT / "reports/causal_preclose/causal_preclose_summary.json"
IBS_RESULT = ROOT / "reports/ab_fixed_priority_validation.json"
IBS_COMBINED = ROOT / "reports/ab_fixed_priority_combined_daily.csv"
A_ARTIFACT_NAME = "zarattini-2019-2025-clean-baseline-20260928-01"
OUTPUT_DAILY = "ab_fixed_priority_combined_daily.csv"
OUTPUT_SUMMARY = "ab_fixed_priority_validation.json"
EXPECTED_SHA256 = {
    "a_daily": "6415ac0285bf3a4dab438214903b003ff97355972f70e6469d8ed53149f41371",
    "a_trades": "b695520fe07181bd13acaec5d119e333c5b1f005e28f3c4b304cf971936c9b0a",
    "a_manifest": "fbcf25fdc83b14ba4cf54da7521033f0cecd980cae2bcf44737dfe33879ebac8",
    "b_daily": "b15a7850244e31ff79b7a716d7e9373938dada063acde2e17e0f1fb0118c4895",
    "b_summary": "2da69a1ab414d03c890cc869c99997b6bddcd4ce804c82d78997cf8b2657421b",
}
RSI3_ONLY_SHA256 = "f1e4c90460194d28bdf2942f6c3217adfc6a98fc21526c5ac39638f70f8833ab"


def strategy_paths(strategy: str) -> tuple[Path, str, str, str]:
    if strategy == "RSI3_IBS":
        return B_DAILY, OUTPUT_DAILY, OUTPUT_SUMMARY, EXPECTED_SHA256["b_daily"]
    if strategy == "RSI3_ONLY":
        return (B_ROOT / "reports/causal_preclose/RSI3_ONLY_daily.csv",
                "ab_rsi3_only_fixed_priority_combined_daily.csv",
                "ab_rsi3_only_fixed_priority_validation.json", RSI3_ONLY_SHA256)
    raise ValueError(f"Unknown B strategy: {strategy}")


def control_comparison(only: dict, ibs: dict, only_contribution: dict,
                       ibs_contribution: dict) -> dict:
    metric_keys = ("compounded_return", "cagr", "annualized_volatility", "sharpe_rf_0",
                   "max_drawdown", "calmar")
    delta = {key: only[key] - ibs[key] for key in metric_keys if key in only and key in ibs}
    delta.update({key: only_contribution[key] - ibs_contribution[key]
                  for key in only_contribution if key in ibs_contribution
                  and isinstance(only_contribution[key], (int, float))
                  and not isinstance(only_contribution[key], bool)})
    quality_keys = ("cagr", "sharpe_rf_0", "max_drawdown", "calmar")
    if all(abs(delta[key]) < 1e-9 for key in quality_keys):
        verdict = "NO_MATERIAL_DIFFERENCE"
    elif all(delta[key] >= 0 for key in quality_keys):
        verdict = "RSI3_ONLY_COMBINATION_SUPERIOR"
    else:
        verdict = "RSI3_IBS_COMBINATION_PREFERRED"
    return {"verdict": verdict, "delta": delta,
            "delta_convention": "RSI3_ONLY minus RSI3_IBS; positive MDD means shallower drawdown"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decimal(value: str, field: str) -> Decimal:
    try:
        parsed = Decimal(value)
        if parsed.is_finite():
            return parsed
    except (InvalidOperation, TypeError):
        pass
    raise ValueError(f"BLOCKED_B_RETURN_ALIGNMENT: invalid {field}: {value!r}")


def combine_daily(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Select exactly one return per session using B's previous close state."""
    dates = a["date"].tolist()
    if dates != b["date"].tolist() or dates != sorted(set(dates)):
        raise ValueError("SESSION_DATE_MISMATCH")
    positions = b["position"].tolist()
    if any(value not in ("0", "1") for value in positions):
        raise ValueError("BLOCKED_B_RETURN_ALIGNMENT: invalid position")
    if b["strategy_return"].iloc[0] != "" or b["execution_interval_total_return"].iloc[0] != "":
        raise ValueError("BLOCKED_B_RETURN_ALIGNMENT: first execution has no prior interval")

    rows = []
    combined_equity = 1.0
    previous_position = 0
    for i, (a_row, b_row) in enumerate(zip(a.itertuples(index=False), b.itertuples(index=False))):
        current_position = int(b_row.position)
        action = b_row.signal_action
        if i > 0:
            expected_action = (
                "ENTRY" if previous_position == 0 and current_position == 1
                else "EXIT" if previous_position == 1 and current_position == 0
                else "HOLD" if current_position == 1
                else None
            )
            if expected_action is not None and action != expected_action:
                raise ValueError("BLOCKED_B_RETURN_ALIGNMENT: position/action transition")
            if expected_action is None and action not in ("FLAT", "MISSING_RSI", "MISSING_IBS"):
                raise ValueError("BLOCKED_B_RETURN_ALIGNMENT: flat position/action transition")
            previous_price = _decimal(b.iloc[i - 1]["exec_price"], "previous exec_price")
            current_price = _decimal(b_row.exec_price, "exec_price")
            cash = _decimal(b_row.cash_distribution, "cash_distribution")
            if previous_price <= 0 or current_price <= 0:
                raise ValueError("BLOCKED_B_RETURN_ALIGNMENT: nonpositive execution price")
            calculated_interval = (current_price + cash) / previous_price - 1
            recorded_interval = _decimal(b_row.execution_interval_total_return, "interval return")
            recorded_strategy = _decimal(b_row.strategy_return, "strategy return")
            if abs(calculated_interval - recorded_interval) > Decimal("1e-22"):
                raise ValueError("BLOCKED_B_RETURN_ALIGNMENT: execution interval return")
            if abs(recorded_strategy - previous_position * calculated_interval) > Decimal("1e-22"):
                raise ValueError("BLOCKED_B_RETURN_ALIGNMENT: strategy return does not follow prior position")
            b_return = float(recorded_strategy)
        else:
            b_return = 0.0

        a_return = float(a_row.gross_return)
        if not math.isfinite(a_return) or not math.isfinite(b_return):
            raise ValueError("NONFINITE_DAILY_RETURN")
        a_trade_count = int(a_row.trade_count)
        if a_trade_count < 0 or (a_trade_count == 0 and a_return != 0):
            raise ValueError("A_TRADE_RETURN_MISMATCH")
        b_priority = previous_position == 1
        chosen_return = b_return if b_priority else a_return
        start_equity = combined_equity
        combined_equity *= 1 + chosen_return
        rows.append(
            {
                "date": a_row.date,
                "a_gross_return": a_return,
                "a_gross_pnl_baseline_usd": float(a_row.daily_gross_pnl),
                "a_trade_count": a_trade_count,
                "b_position_before_execution": previous_position,
                "b_position_after_execution": current_position,
                "b_signal_action": action,
                "b_strategy_return": b_return,
                "selected_source": "B" if b_priority else "A",
                "combined_return": chosen_return,
                "combined_start_equity": start_equity,
                "combined_end_equity": combined_equity,
                "suppressed_a_trade_count": a_trade_count if b_priority else 0,
                "suppressed_a_gross_pnl_baseline_usd": float(a_row.daily_gross_pnl) if b_priority else 0.0,
                "suppressed_a_return": a_return if b_priority else 0.0,
                "retained_a_gross_pnl_baseline_usd": float(a_row.daily_gross_pnl) if not b_priority else 0.0,
                "retained_a_return": a_return if not b_priority else 0.0,
                "a_combined_equity_gain": start_equity * a_return if not b_priority else 0.0,
                "b_combined_equity_gain": start_equity * b_return if b_priority else 0.0,
                "suppressed_a_counterfactual_gain_on_combined_capital": (
                    start_equity * a_return if b_priority else 0.0
                ),
            }
        )
        previous_position = current_position
    return pd.DataFrame(rows)


def performance(dates: list[str], returns: list[float]) -> dict:
    """Use B's published convention on a common session calendar."""
    equity = []
    value = 1.0
    annual_growth: dict[str, float] = {}
    for day, daily_return in zip(dates, returns):
        value *= 1 + daily_return
        equity.append(value)
        annual_growth[day[:4]] = annual_growth.get(day[:4], 1.0) * (1 + daily_return)
    years = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days / 365.25
    cagr = value ** (1 / years) - 1
    sample = returns[1:]  # first session has no preceding execution interval
    stdev = statistics.stdev(sample)
    volatility = stdev * math.sqrt(252)
    sharpe = statistics.mean(sample) / stdev * math.sqrt(252) if stdev else None
    peak = 1.0
    mdd = 0.0
    for value_at_close in equity:
        peak = max(peak, value_at_close)
        mdd = min(mdd, value_at_close / peak - 1)
    return {
        "compounded_return": equity[-1] - 1,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_rf_0": sharpe,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else None,
        "yearly_returns": {year: growth - 1 for year, growth in annual_growth.items()},
    }


def _artifact_dir() -> Path:
    linux = Path("/home/lidong/temp_file") / A_ARTIFACT_NAME
    windows = Path("\\\\wsl.localhost\\Ubuntu\\home\\lidong\\temp_file") / A_ARTIFACT_NAME
    for candidate in (linux, windows):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Authoritative A artifact directory unavailable; pass --a-artifact-dir")


def _verify_a_trades(a: pd.DataFrame, trades: pd.DataFrame) -> None:
    entries = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    exits = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    if not entries.equals(exits) or not set(exits).issubset(set(a["date"])):
        raise ValueError("A_TRADE_SESSION_MAPPING_MISMATCH")
    counts = exits.value_counts().reindex(a["date"], fill_value=0).to_numpy()
    if not np.array_equal(counts, a["trade_count"].to_numpy()) or len(trades) != 1509:
        raise ValueError("A_TRADE_COUNT_MISMATCH")


def _verify_standalones(a: pd.DataFrame, b: pd.DataFrame, a_manifest: dict, b_summary: dict,
                        metrics: dict, b_strategy: str) -> dict:
    a_net = a["net_return"].to_numpy(dtype=float)
    a_net_equity = 100000.0 * np.cumprod(1 + a_net)
    a_net_rebuild_error = float(np.max(np.abs(a_net_equity - a["end_net_equity"].to_numpy(dtype=float))))
    original = a_manifest["clean_baseline"]
    if a_net_rebuild_error > 1e-7 or abs(a_net_equity[-1] - original["final_equity"]) > 1e-7:
        raise ValueError("A_BASELINE_NET_IDENTITY_MISMATCH")
    a_net_cagr_252 = a_net_equity[-1] / 100000.0
    a_net_cagr_252 = a_net_cagr_252 ** (252 / len(a)) - 1
    a_net_sharpe = float(a_net.mean() / a_net.std(ddof=1) * math.sqrt(252))
    a_net_mdd = float(np.min(a_net_equity / np.maximum.accumulate(a_net_equity) - 1))
    if (abs(a_net_cagr_252 - original["cagr"]) > 1e-10
            or abs(a_net_sharpe - original["sharpe"]) > 1e-10
            or abs(a_net_mdd - original["max_drawdown"]) > 1e-10):
        raise ValueError("A_BASELINE_NET_METRIC_MISMATCH")

    b_equity = np.cumprod(1 + b["strategy_return"].replace("", "0").astype(float).to_numpy())
    b_equity_error = float(np.max(np.abs(b_equity - b["equity"].astype(float).to_numpy())))
    if b_equity_error > 1e-12:
        raise ValueError("B_BASELINE_EQUITY_MISMATCH")
    published = b_summary["strategies"][b_strategy]
    b_metric = metrics["B"]
    for source_key, reported_key in (
        ("compounded_return", "compounded_return_pct"),
        ("cagr", "cagr_pct"),
        ("annualized_volatility", "annualized_volatility_pct"),
        ("max_drawdown", "max_drawdown_pct"),
    ):
        if abs(100 * b_metric[source_key] - published[reported_key]) > 1e-8:
            raise ValueError(f"B_BASELINE_METRIC_MISMATCH: {source_key}")
    for key, reported_key in (("sharpe_rf_0", "sharpe_rf_0"), ("calmar", "calmar")):
        if abs(b_metric[key] - published[reported_key]) > 1e-9:
            raise ValueError(f"B_BASELINE_METRIC_MISMATCH: {key}")
    for year, value in b_metric["yearly_returns"].items():
        if abs(100 * value - published["yearly_compounded_returns_pct"][year]) > 1e-8:
            raise ValueError(f"B_BASELINE_YEAR_MISMATCH: {year}")
    b_exposure = sum(b["position"].iloc[:-1].astype(int)) / (len(b) - 1)
    b_completed_trades = int(b["signal_action"].eq("EXIT").sum())
    if (abs(100 * b_exposure - published["exposure_pct"]) > 1e-8
            or b_completed_trades != published["completed_trades"]):
        raise ValueError("B_BASELINE_EXPOSURE_OR_TRADES_MISMATCH")
    return {
        "a_net_equity_max_rebuild_error_usd": a_net_rebuild_error,
        "a_net_cagr_252_session_convention": a_net_cagr_252,
        "a_net_sharpe": a_net_sharpe,
        "a_net_max_drawdown": a_net_mdd,
        "b_equity_max_rebuild_error_normalized": b_equity_error,
        "b_published_metrics_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-artifact-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--b-strategy", choices=("RSI3_IBS", "RSI3_ONLY"), default="RSI3_IBS")
    args = parser.parse_args()
    artifact = args.a_artifact_dir or _artifact_dir()
    b_daily, output_daily, output_summary, b_expected_hash = strategy_paths(args.b_strategy)
    paths = {
        "a_daily": A_DAILY,
        "a_trades": artifact / "trades_clean_baseline.csv",
        "a_manifest": artifact / "clean_baseline_2019_2025.json",
        "b_daily": b_daily,
        "b_summary": B_SUMMARY,
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    expected_hashes = {**EXPECTED_SHA256, "b_daily": b_expected_hash}
    for name, expected in expected_hashes.items():
        if hashes[name] != expected:
            raise ValueError(f"BLOCKED_INPUT_IDENTITY: {name}: {hashes[name]}")
    a = pd.read_csv(A_DAILY)
    b = pd.read_csv(b_daily, dtype=str, keep_default_na=False)
    a_manifest = json.loads(paths["a_manifest"].read_text(encoding="utf-8"))
    b_summary = json.loads(B_SUMMARY.read_text(encoding="utf-8"))
    if (len(a) != 1760 or len(b) != 1760 or a["date"].iloc[0] != "2019-01-02"
            or a["date"].iloc[-1] != "2025-12-31"
            or b_summary["sample"]["sessions"] != 1760
            or a_manifest["trading_days_corrected"] != 1760):
        raise ValueError("BLOCKED_INPUT_IDENTITY: sample")
    _verify_a_trades(a, pd.read_csv(paths["a_trades"]))
    combined = combine_daily(a, b)
    dates = combined["date"].tolist()
    results = {
        "A": performance(dates, a["gross_return"].astype(float).tolist()),
        "B": performance(dates, combined["b_strategy_return"].tolist()),
        "AB": performance(dates, combined["combined_return"].tolist()),
    }
    standalone_checks = _verify_standalones(a, b, a_manifest, b_summary, results, args.b_strategy)
    rebuilt_equity = np.cumprod(1 + combined["combined_return"].to_numpy(dtype=float))
    combined_rebuild_error = float(np.max(np.abs(rebuilt_equity - combined["combined_end_equity"].to_numpy())))
    if combined_rebuild_error > 1e-12:
        raise ValueError("COMBINED_LEDGER_RECONSTRUCTION_FAILED")
    priority = combined["selected_source"].eq("B")
    raw_a_active = combined["a_trade_count"].gt(0)
    normalized_a_gain = math.fsum(combined["a_combined_equity_gain"])
    normalized_b_gain = math.fsum(combined["b_combined_equity_gain"])
    if abs(normalized_a_gain + normalized_b_gain - combined["combined_end_equity"].iloc[-1] + 1) > 1e-12:
        raise ValueError("COMPONENT_CONTRIBUTION_MISMATCH")
    contribution = {
        "b_priority_sessions": int(priority.sum()),
        "suppressed_a_sessions": int(priority.sum()),
        "suppressed_a_active_sessions": int((priority & raw_a_active).sum()),
        "suppressed_a_trades": int(combined["suppressed_a_trade_count"].sum()),
        "suppressed_a_gross_pnl_baseline_usd": float(combined["suppressed_a_gross_pnl_baseline_usd"].sum()),
        "suppressed_a_arithmetic_gross_return_sum": float(combined["suppressed_a_return"].sum()),
        "suppressed_a_counterfactual_gain_on_combined_capital": float(
            combined["suppressed_a_counterfactual_gain_on_combined_capital"].sum()
        ),
        "retained_a_gross_pnl_baseline_usd": float(combined["retained_a_gross_pnl_baseline_usd"].sum()),
        "retained_a_arithmetic_gross_return_sum": float(combined["retained_a_return"].sum()),
        "retained_a_gain_on_combined_capital": normalized_a_gain,
        "b_arithmetic_return_sum": float(combined.loc[priority, "b_strategy_return"].sum()),
        "b_gain_on_combined_capital": normalized_b_gain,
        "raw_a_trade_day_b_carry_overlap_sessions": int((priority & raw_a_active).sum()),
        "realized_double_count_sessions": 0,
        "a_trade_days_on_b_entry_sessions": int(
            (raw_a_active & combined["b_signal_action"].eq("ENTRY")).sum()
        ),
    }
    ab = results["AB"]
    a_result = results["A"]
    if ab["cagr"] >= 0.30 and ab["max_drawdown"] > -0.10:
        verdict = "AB_STRUCTURAL_TARGET_MET"
    elif (ab["cagr"] > a_result["cagr"] and ab["sharpe_rf_0"] > a_result["sharpe_rf_0"]
          and ab["calmar"] > a_result["calmar"] and ab["max_drawdown"] > a_result["max_drawdown"]):
        verdict = "AB_STRUCTURAL_INCREMENTAL_BUT_TARGET_NOT_MET"
    else:
        verdict = "B_NOT_INCREMENTAL"

    comparison = None
    if args.b_strategy == "RSI3_ONLY":
        authoritative = json.loads(IBS_RESULT.read_text(encoding="utf-8"))
        if (authoritative["input_sha256"] != EXPECTED_SHA256
                or sha256(IBS_COMBINED) != authoritative["combined_daily_sha256"]
                or authoritative["metrics"]["A"] != results["A"]):
            raise ValueError("BLOCKED_INPUT_IDENTITY: authoritative IBS comparison")
        comparison = control_comparison(results["AB"], authoritative["metrics"]["AB"],
                                        contribution, authoritative["contribution"])
        verdict = comparison["verdict"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / output_daily
    combined.to_csv(output, index=False, float_format="%.17g", lineterminator="\n")
    if args.b_strategy == "RSI3_ONLY":
        serialized = pd.read_csv(output)
        rebuilt_metrics = performance(serialized["date"].tolist(),
                                      serialized["combined_return"].tolist())
        for key in ("compounded_return", "cagr", "annualized_volatility",
                    "sharpe_rf_0", "max_drawdown", "calmar"):
            if abs(rebuilt_metrics[key] - results["AB"][key]) > 1e-12:
                raise ValueError(f"COMBINED_LEDGER_METRIC_RECONSTRUCTION_FAILED: {key}")
        for year, value in rebuilt_metrics["yearly_returns"].items():
            if abs(value - results["AB"]["yearly_returns"][year]) > 1e-12:
                raise ValueError(f"COMBINED_LEDGER_YEAR_RECONSTRUCTION_FAILED: {year}")
    summary = {
        "verdict": verdict,
        "input_sha256": hashes,
        "alignment": "position_(t-1) earns execution_(t-1) to execution_t return on date t; position_t is set at execution_t",
        "metrics_convention": "calendar-day CAGR from first to last session; sample daily std and Sharpe on sessions 2..1760, annualized by 252; RF=0",
        "metrics": results,
        "contribution": contribution,
        "validation": {
            **standalone_checks,
            "sessions": len(combined),
            "dates_unique_sorted_and_matched": True,
            "a_trade_counts_match_authoritative_trade_ledger": True,
            "b_return_matches_prior_position_and_execution_prices": True,
            "one_selected_source_per_session": True,
            "combined_equity_max_rebuild_error_normalized": combined_rebuild_error,
            "normalized_component_gain_identity_error": abs(
                normalized_a_gain + normalized_b_gain - combined["combined_end_equity"].iloc[-1] + 1
            ),
        },
        "combined_daily_sha256": sha256(output),
    }
    if args.b_strategy == "RSI3_ONLY":
        summary.update({
            "b_strategy": args.b_strategy,
            "control_comparison": comparison,
            "target_met": ab["cagr"] >= 0.30 and ab["max_drawdown"] > -0.10,
            "next_step": ("COMBINED_COST_VALIDATION" if
                          (ab["cagr"] >= 0.30 and ab["max_drawdown"] > -0.10)
                          else "SEARCH_STRATEGY_C"),
        })
        summary["validation"]["serialized_combined_ledger_rebuilds_all_metrics"] = True
    summary_path = args.out_dir / output_summary
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "combined_daily": str(output),
                      "combined_daily_sha256": summary["combined_daily_sha256"],
                      "summary": str(summary_path), "metrics": results,
                      "contribution": contribution}, indent=2))


if __name__ == "__main__":
    main()
