# Hybrid v18 RSS experiments

These are isolated research candidates based on the unchanged
`team_agent_hybrid_receiver_overflow_v17` tree. They are not promoted agents;
neither has a valid RSS or CPU result yet.

## Candidates

| Folder | Single added idea | Evaluation baseline |
| --- | --- | --- |
| `agents/team_agent_hybrid_risk_gate_v18` | Invoke v17's need-priority repair only when observed route closures, route prohibitions, or sink backlog indicate stress. | `team_agent_hybrid_receiver_overflow_v17` |
| `agents/team_agent_hybrid_shadow_price_v18` | Use an LP over the existing heuristic requests to allocate scarce source stock, entry-edge capacity, and fleet by shortage value; use LP duals as resource prices. | `team_agent_hybrid_receiver_overflow_v17` |

Both candidates keep the heuristic policy and the v17 clipping-aware repair.
The repair projects requests through the simulator's edge, source-stock, and
fleet clipping order, then uses one ledger for reservations. The third idea in
`additional_tasks.md` is therefore already present in v17 and is reused as the
common baseline rather than implemented a second time.

## Implementation notes

The risk gate only reads observed fields. It does not interpret raw warnings
as probabilities. Missing or malformed signals do not trigger the specialist;
positive observed sink backlog can trigger it directly. Its closure and blocked
route thresholds are parameters in the candidate's `params.json`.

The shadow-price candidate only redistributes amounts requested by the
heuristic when the common ledger predicts clipping. It values a need using its
dated shortage penalty and route cost, omits unknown or nonpositive values, and
falls back to v17's greedy priority if SciPy's LP is unavailable, fails, or the
candidate set exceeds 2,048 variables. The LP's source, edge, and fleet duals
are recorded as USD per capacity unit. Traces are capped for prices and
assignments; the complete allocation is still applied. Destination storage
safety remains supplied by v17's receiver-overflow adjustment; the LP cannot
increase a slot above the heuristic's requested amount.

## Status and evaluation

Status: **unvalidated experiment**. The local environment does not currently
provide a usable `uv`/`sbf` runtime, so no paired RSS, per-harm, cost-component,
clipping, or CPU result is claimed.
Candidate tree and changed-file fingerprints are recorded in
[`TEAM_AGENT_HYBRID_V18_EXPERIMENTS.json`](TEAM_AGENT_HYBRID_V18_EXPERIMENTS.json).

Follow `TO_DO.md` before considering either candidate for promotion:

1. Compare each candidate against unchanged v17 on root `12345`, at least 64
   Full episodes.
2. Independently compare on root `67890`, at least 64 Full episodes, and 64
   Small episodes. Use root `0` only for final confirmation.
3. Run `sbf check` for Small and Full, and inspect paired intervals, harm-level
   RSS, unmet critical demand, freight/tariff/holding cost, clipping, causal
   traces, and worst-case CPU.
4. Keep each experiment separate. Do not combine the risk gate and shadow
   prices until each has independent evidence of benefit.

Suggested paired runs:

```sh
uv run sbf compare team_agent_hybrid_risk_gate_v18 team_agent_hybrid_receiver_overflow_v17 --task=full --episodes=64 --entropy=12345 --cpu_budget=True
uv run sbf compare team_agent_hybrid_shadow_price_v18 team_agent_hybrid_receiver_overflow_v17 --task=full --episodes=64 --entropy=12345 --cpu_budget=True
```
