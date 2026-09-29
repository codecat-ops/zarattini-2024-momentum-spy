# Ex-ante protocol — Databento VWAP forward challenger

**Freeze:** the first Git commit adding this file. Its commit time, converted to
`America/New_York`, is the sole time anchor. This protocol must be committed
before any eligible session. The 2019–2025 clean Databento sample and any other
session on or before the freeze date are historical evidence, never forward
confirmation data. No interim performance review or rule change is allowed.

## Pair and input

- Authoritative control: `exit_mode=final`. Sole challenger: `exit_mode=vwap`.
  Use strategy code commit `01980fd9c0f4664d5de038989cbff4a1be9d8670`.
- Feed both runs the **same** SPY Databento 1-minute RTH bars, with the same
  NYSE session boundaries and early-close cleaning as the authoritative clean
  Baseline. Save the forward input and its SHA-256 at evaluation. A missing or
  corrupt session is repaired using market-data evidence before either result
  is examined; never substitute a different session or extend the window.
- Explicitly pass for **both** runs: initial equity `$100,000`, lookback `14`,
  check interval `30` minutes, VM `1.0`, daily vol target `0.02`, max leverage
  `4.0`, vol lookback `14`, stock unit multiplier `1`, commission `$0.0035`
  per share with `$0.35` minimum per order, fees `0`, slippage `0`.
  Signals, sizing, order timing, forced close, and all other inputs are identical.
  Only `exit_mode` differs. Both accounts start flat at `$100,000`.

## One observation window and one decision

The first actual NYSE RTH session whose New York calendar date is **strictly
after** the freeze commit's New York date is forward session 1. Sessions 1–15
are indicator warmup only: no counted trades or returns. Sessions 16–267 are
the **252-session evaluation window**. An early-close session counts as one
session. The close of session 267 is the only decision point; trade count,
interim results, calendar-year boundaries, and statistical significance do
not move it. Pre-freeze bars cannot contribute trades or evaluation returns.
If the 267-session input cannot be made complete and paired at that point,
record `KEEP_CHALLENGER` for insufficient data; do not extend or reselect days.

## Net metrics and mechanical verdict

For each evaluation day, use end-of-day equity **after all stated costs** to
calculate `r_final` and `r_vwap` from the preceding day's equity, then
`d = r_vwap - r_final` on the same 252 days. Report mean `d`, `252 × mean(d)`,
and a two-sided 95% Newey–West interval for mean `d` (Bartlett weights, fixed
lag 5, autocovariances divided by 252, normal multiplier 1.96). Report net
terminal equity, annualized daily Sharpe and daily-equity MDD for each run,
using the definitions in `src/stats.py:performance_summary`. A higher MDD
(less negative) is better. Keep full precision for decisions.

At the single decision point, apply these rules **in order**:

1. `PROMOTE` if the paired interval's lower bound is `> 0`, VWAP net terminal
   equity is `> final`, and **either** VWAP Sharpe is `>= final` **or** VWAP MDD
   is `>= final`.
2. `REJECT` if the paired interval's upper bound is `< 0`; or if VWAP net
   terminal equity is `<= final` **and** VWAP Sharpe is `<= final` **and** VWAP
   MDD is `<= final`.
3. Otherwise `KEEP_CHALLENGER`. Undefined metrics also yield
   `KEEP_CHALLENGER`.

`final` remains the authoritative Baseline throughout observation. Promotion
can occur only after this one verdict. `KEEP_CHALLENGER` does not extend this
window or authorize a second look under this protocol. Do not add NR4, other
exit modes, lookbacks, intervals, VM values, or sizing variants.
