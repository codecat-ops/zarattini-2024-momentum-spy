r"""Materialize Databento baseline daily gross returns without rerunning the strategy.

Example (Windows, with the authoritative WSL files mounted):
    python scripts/materialize_databento_gross_daily.py \
      --artifact-dir '\\wsl.localhost\Ubuntu\home\lidong\temp_file\zarattini-2019-2025-clean-baseline-20260928-01' \
      --source-csv '\\wsl.localhost\Ubuntu\home\lidong\EquityMind\data\USAStock\SPY\canonical\2019-2025\databento\SPY_1m_regular_session_databento_canonical_candidate_v1.csv'

The trade sizes and daily capital base are those of the realized net-equity
baseline. A compounded gross-return index would not be a strategy rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


INITIAL_EQUITY = 100_000.0
EXPECTED_SOURCE = (
    "/home/lidong/EquityMind/data/USAStock/SPY/canonical/2019-2025/databento/"
    "SPY_1m_regular_session_databento_canonical_candidate_v1.csv"
)
EXPECTED_SHA256 = {
    "source_csv": "1767b1c7055bf43fc3958a1e57ae346a645a9915149bddd443772977e66a1993",
    "manifest": "fbcf25fdc83b14ba4cf54da7521033f0cecd980cae2bcf44737dfe33879ebac8",
    "cleaned_parquet": "22937752eb5ea4ddcd79a7ae3eedc9bdabde5d5bffe3bc4309353de1de7e5f90",
    "net_daily_equity": "92dcda900e99293b02ba145eb4e03cc251e6ef7e24fd5fb2ae7fb522e967df32",
    "gross_trade_ledger": "b695520fe07181bd13acaec5d119e333c5b1f005e28f3c4b304cf971936c9b0a",
}
LEDGER_NAME = "databento_2019_2025_baseline_daily_gross.csv"
SUMMARY_NAME = "databento_2019_2025_baseline_ledger_validation.json"
ET = ZoneInfo("America/New_York")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_date(value: object) -> str:
    """Require an explicit, correct New York UTC offset on artifact timestamps."""
    try:
        timestamp = datetime.fromisoformat(str(value))
        if timestamp.tzinfo is None:
            raise ValueError("missing UTC offset")
        local = timestamp.astimezone(ET)
        if timestamp.date() != local.date() or timestamp.utcoffset() != local.utcoffset():
            raise ValueError("timestamp is not expressed in New York time")
        return local.date().isoformat()
    except (TypeError, ValueError) as error:
        raise ValueError(f"BLOCKED_TRADE_SESSION_MAPPING: {value!r}") from error


def build_daily_ledger(
    daily_equity: pd.DataFrame,
    trades: pd.DataFrame,
    session_dates: pd.Index,
    initial_equity: float = INITIAL_EQUITY,
) -> tuple[pd.DataFrame, dict]:
    """Join existing net equity and trade P&L to the cleaned input sessions."""
    sessions = list(session_dates)
    if not sessions or sessions != sorted(set(sessions)):
        raise ValueError("SESSION_COVERAGE_MISMATCH: source sessions are not unique and sorted")
    daily_dates = [_session_date(value) for value in daily_equity["date"]]
    if daily_dates != sessions:
        raise ValueError("SESSION_COVERAGE_MISMATCH: net daily dates differ from cleaned input")

    entries = [_session_date(value) for value in trades["entry_time"]]
    exits = [_session_date(value) for value in trades["exit_time"]]
    if any(entry != exit for entry, exit in zip(entries, exits)):
        raise ValueError("BLOCKED_OVERNIGHT_POSITION_FOUND")
    if not set(exits).issubset(set(sessions)):
        raise ValueError("BLOCKED_TRADE_SESSION_MAPPING: trade date is not a baseline session")

    for column in ("gross_pnl", "costs", "net_pnl"):
        if not pd.api.types.is_numeric_dtype(trades[column]) or not trades[column].map(math.isfinite).all():
            raise ValueError(f"TRADE_PNL_INVALID: {column}")
    if not daily_equity["equity"].map(math.isfinite).all():
        raise ValueError("DAILY_EQUITY_INVALID")

    trade_identity_error = float(
        (trades["gross_pnl"] - trades["costs"] - trades["net_pnl"]).abs().max()
    )
    if trade_identity_error > 1e-8:
        raise ValueError("TRADE_PNL_INCONSISTENT")

    by_day: dict[str, list[int]] = {day: [] for day in sessions}
    for index, day in enumerate(exits):
        by_day[day].append(index)
    if sum(map(len, by_day.values())) != len(trades):
        raise ValueError("BLOCKED_TRADE_SESSION_MAPPING: trade assigned more or less than once")

    rows = []
    max_daily_net_error = 0.0
    max_daily_trade_identity_error = 0.0
    max_compounded_net_error = 0.0
    compounded_net = initial_equity
    previous_net_equity = initial_equity
    for day, end_equity in zip(sessions, daily_equity["equity"]):
        indexes = by_day[day]
        gross = math.fsum(float(trades.iloc[i]["gross_pnl"]) for i in indexes)
        costs = math.fsum(float(trades.iloc[i]["costs"]) for i in indexes)
        net = math.fsum(float(trades.iloc[i]["net_pnl"]) for i in indexes)
        start = previous_net_equity
        end = float(end_equity)
        if start <= 0 or not math.isfinite(end):
            raise ValueError("DAILY_EQUITY_INVALID")
        max_daily_trade_identity_error = max(max_daily_trade_identity_error, abs(gross - costs - net))
        max_daily_net_error = max(max_daily_net_error, abs(end - start - net))
        gross_return = gross / start
        net_return = net / start
        compounded_net *= 1.0 + net_return
        max_compounded_net_error = max(max_compounded_net_error, abs(compounded_net - end))
        rows.append(
            {
                "date": day,
                "start_equity": start,
                "daily_gross_pnl": gross,
                "daily_costs": costs,
                "daily_net_pnl": net,
                "gross_return": gross_return,
                "net_return": net_return,
                "end_net_equity": end,
                "trade_count": len(indexes),
            }
        )
        previous_net_equity = end

    if max_daily_trade_identity_error > 1e-8 or max_daily_net_error > 1e-7:
        raise ValueError("DAILY_PNL_RECONCILIATION_FAILED")
    if max_compounded_net_error > 1e-7:
        raise ValueError("NET_RETURN_RECONSTRUCTION_FAILED")

    ledger = pd.DataFrame(rows)
    totals = {
        "gross_pnl": math.fsum(ledger["daily_gross_pnl"]),
        "costs": math.fsum(ledger["daily_costs"]),
        "net_pnl": math.fsum(ledger["daily_net_pnl"]),
    }
    trade_totals = {
        "gross_pnl": math.fsum(trades["gross_pnl"]),
        "costs": math.fsum(trades["costs"]),
        "net_pnl": math.fsum(trades["net_pnl"]),
    }
    if any(abs(totals[key] - trade_totals[key]) > 1e-8 for key in totals):
        raise ValueError("TRADE_DAILY_TOTAL_MISMATCH")
    checks = {
        "session_count": len(sessions),
        "trade_count": len(trades),
        "no_trade_sessions": int((ledger["trade_count"] == 0).sum()),
        "all_trades_intraday": True,
        "all_trades_assigned_once": True,
        "daily_dates_match_cleaned_input": True,
        "max_trade_gross_minus_costs_minus_net_error": trade_identity_error,
        "max_daily_gross_minus_costs_minus_net_error": max_daily_trade_identity_error,
        "max_daily_net_equity_delta_error": max_daily_net_error,
        "max_compounded_net_equity_error": max_compounded_net_error,
        "totals": totals,
        "trade_totals": trade_totals,
    }
    return ledger, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports")
    args = parser.parse_args()

    root = args.artifact_dir
    paths = {
        "source_csv": args.source_csv,
        "manifest": root / "clean_baseline_2019_2025.json",
        "cleaned_parquet": root / "SPY_1m_regular_session_nyse_boundary_clean_2019_2025.parquet",
        "net_daily_equity": root / "daily_equity_clean_baseline.csv",
        "gross_trade_ledger": root / "trades_clean_baseline.csv",
    }
    hashes = {key: sha256(path) for key, path in paths.items()}
    for key, expected in EXPECTED_SHA256.items():
        if hashes[key] != expected:
            raise ValueError(f"BASELINE_ARTIFACT_IDENTITY_MISMATCH: {key}: {hashes[key]}")

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest["source"] != EXPECTED_SOURCE:
        raise ValueError("BASELINE_ARTIFACT_IDENTITY_MISMATCH: manifest source")
    baseline = manifest["clean_baseline"]
    if baseline["trade_count"] != 1509 or manifest["trading_days_corrected"] != 1760:
        raise ValueError("BASELINE_ARTIFACT_IDENTITY_MISMATCH: manifest counts")

    clean = pd.read_parquet(paths["cleaned_parquet"], columns=["timestamp_utc"])
    if len(clean) != manifest["corrected_row_count"]:
        raise ValueError("BASELINE_ARTIFACT_IDENTITY_MISMATCH: cleaned bar count")
    session_index = pd.to_datetime(clean["timestamp_utc"], utc=True).dt.tz_convert(ET)
    session_dates = pd.Index(session_index.dt.strftime("%Y-%m-%d").unique())
    daily = pd.read_csv(paths["net_daily_equity"]).rename(columns={"Unnamed: 0": "date"})
    trades = pd.read_csv(paths["gross_trade_ledger"])
    ledger, checks = build_daily_ledger(daily, trades, session_dates, INITIAL_EQUITY)
    if checks["session_count"] != 1760 or checks["trade_count"] != 1509:
        raise ValueError("BASELINE_ARTIFACT_IDENTITY_MISMATCH: ledger counts")
    if abs(checks["totals"]["costs"] - baseline["total_recorded_transaction_costs"]) > 1e-8:
        raise ValueError("BASELINE_ARTIFACT_IDENTITY_MISMATCH: costs")
    if abs(ledger["end_net_equity"].iloc[-1] - baseline["final_equity"]) > 1e-8:
        raise ValueError("BASELINE_ARTIFACT_IDENTITY_MISMATCH: final equity")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.out_dir / LEDGER_NAME
    ledger.to_csv(ledger_path, index=False, float_format="%.17g", lineterminator="\n")
    ledger_hash = sha256(ledger_path)
    summary = {
        "verdict": "NOISE_AREA_GROSS_DAILY_LEDGER_READY",
        "gross_return_semantics": "pre-cost gross P&L / prior-session realized net equity; actual net-equity sizing path",
        "input_sha256": hashes,
        "ledger_sha256": ledger_hash,
        "checks": checks,
    }
    summary_path = args.out_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ledger": str(ledger_path), "ledger_sha256": ledger_hash, "summary": str(summary_path), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
