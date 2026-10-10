# Hybrid RSS v25 candidate

`agents/team_agent_hybrid_rss_v25` combines implementation features and
selected settings from the v19–v24 experiment folders. It is an unvalidated
integration candidate; there is no score evidence that the combined policy
improves RSS.

## Integrated work

- **v19 clipping recovery:** start from the projected action, keep its executed
  flows protected, and recover only clipped quantities that have a positive
  dated shortage value and an observed, legal, feasible route. A shared ledger
  reserves source stock, entry capacity, and fleet. The receiver-overflow
  policy is reapplied to proposed additions; recovery is rejected if it would
  cut any already-protected flow.
- **v20 fuel experiments:** enable receiver cover 8, industry bonus 2, fuel
  lookahead horizon 4, and fuel MPC horizon 12.
- **v21 production experiments:** use production balance power 1, disable the
  production-output weighting, set recovery cover 8, and enable production
  energy/value weights at 1.
- **v22 backlog valuation:** add backlog-aware production value with strength 1.
- **v23 production network:** retain the wafer-network model at strength 0.5.
- **v24 route probes:** use soft closure power 0.5, the lower tested cost-bias
  setting of 1, and disable route-warning bias. The mutually exclusive cost
  sweep settings were not combined.

The full allocator remains disabled. All these additions are opt-in settings
in v25; earlier agent folders are unchanged. The merge is intentionally a
single integration experiment, so an eventual RSS change cannot identify the
contribution of each component without follow-up ablations.

## Evaluation status

No v19–v24 paired score reports are present in the repository. This environment
also has no usable `uv` or `sbf` command or installed Python runtime. Therefore
v25 has not been run, timed, or score-validated; **an RSS improvement has not
been established**. The combination of fuel MPC, fuel lookahead, and the
production-network model especially needs worst-case CPU checks.

Use `TO_DO.md`'s protocol before promotion: at least 64 Full episodes on root
12345, independent 64-episode Full and Small comparisons on root 67890, root 0
for final confirmation only, and `sbf check` for Small and Full. Compare first
with v23 to measure the merged additions, and retain a fixed v15 or v16
reference for the project-level RSS target. Review paired intervals, per-harm
RSS, shortage and shipping costs, clipping/recovery traces, receiver storage,
and CPU time.
