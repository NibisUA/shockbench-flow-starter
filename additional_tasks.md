# Additional RSS Improvement Tasks

## Prompt 1: Risk-gated policy portfolio

```text
Work in the ShockBench-Flow project. Read AGENTS.md, PROJECT_CONTEXT.md, and TO_DO.md before changing code. Develop one isolated agent candidate for a risk-gated policy portfolio.

Start from agents/team_agent_hybrid_receiver_overflow_v17 as the experimental baseline. Do not modify or overwrite v15, v16, v17, frozen artifacts, or existing benchmark results. Create a new agent folder with a distinct versioned name.

Idea:
Keep the existing heuristic as the default policy in calm conditions. Switch to a disruption specialist only when information available in the current observation indicates that the specialist’s expected savings exceed its extra freight, tariff, holding, and inventory costs. Use observable disruptions, lead-time-adjusted shortages, and calibrated warning evidence. Do not treat raw warning scores as probabilities, use hidden scenario data at runtime, or gate on episode IDs or seeds.

Before changing policy:
1. Record the current agent files and parameters with hashes.
2. Run and save a paired baseline report with per-harm RSS and cost components.
3. Capture causal traces showing which needs, routes, and resource limits drive the decision.
4. If the benchmark runtime is unavailable, document the exact blocker and leave the candidate explicitly unvalidated.

Implementation:
- Make the gate deterministic and dependent only on current observation/configuration.
- Preserve the baseline action when the gate is uncertain or required evidence is missing.
- Ensure both policies use the same observation and do not double-count inventory, arrivals, or shared capacity.
- Keep the agent self-contained and within the submission import, file, and CPU limits.
- Expose the gate’s decision and its evidence in bounded diagnostics suitable for offline evaluation.

Evaluation:
Follow TO_DO.md: tune on root 12345 with at least 64 Full episodes; compare against the unchanged v17 baseline on identical episodes; validate on root 67890 with at least 64 Full episodes and on Small; use root 0 only for final confirmation. Check per-harm RSS, paired RSS difference and interval, cost components, switch rate, fallback weeks, clipping, and CPU. Run sbf check for Small and Full. Do not claim an improvement or promote the candidate unless the RSS and operational gates pass.
```

## Prompt 2: Dynamic prices for scarce resources

```text
Work in the ShockBench-Flow project. Read AGENTS.md, PROJECT_CONTEXT.md, and TO_DO.md. Build one isolated candidate that allocates scarce resources using dynamic shadow prices.

Start from agents/team_agent_hybrid_receiver_overflow_v17, preserve it unchanged, and create a new versioned agent folder. Do not enable the full allocator pipeline as the production path.

Idea:
When several needs compete for limited source stock, route capacity, fleet, or terminal space, assign a changing price to each scarce resource. Give each feasible shipment a value based on the shortage cost it is expected to avoid by its due week, minus freight, tariffs, holding, and excess-inventory costs. Use those values and resource prices to decide which shipments get the constrained capacity. Unknown or physically late arrivals must not count as timely shortage prevention.

Before changing policy:
- Freeze hashes and parameters for the baseline.
- Capture traces for needs, feasible routes, shared resources, and baseline action clipping.
- Verify units and dated inventory netting, especially for wafer/BOM and the fuel terminal chain.
- Document any missing runtime or data needed to complete these checks.

Implementation:
- Integrate the pricing layer after the existing heuristic has produced its requests.
- Use one shared ledger for source stock, edges, fleet, and arrivals; no stage may reserve the same resource twice.
- Preserve baseline actions when constraints are not binding or the estimated marginal value is unknown.
- Keep the first experiment to one change: do not simultaneously tune warning behavior, forecast horizons, and cover parameters.
- Keep execution within the per-week CPU budget and use only allowed submission dependencies.

Evaluation:
Use paired comparisons against unchanged v17 following TO_DO.md: tune on root 12345, independently validate on root 67890, then confirm on root 0 and check Small. Report Full and Small RSS, per-harm scores, paired intervals, unmet critical demand, freight/tariff/holding costs, changed actions, resource utilization, clipping, and CPU. Do not accept the candidate based only on more shipments or lower cost on one episode. Promote only if it clears the documented RSS, causal, and operational gates.
```

## Prompt 3: Clipping-aware action repair

```text
Work in the ShockBench-Flow project. Read AGENTS.md, PROJECT_CONTEXT.md, TO_DO.md, and the tests/docs describing action clipping and shared-resource limits. Create a new versioned candidate based on agents/team_agent_hybrid_receiver_overflow_v17; do not alter existing baselines or candidates.

Idea:
Repair a proposed action before the simulator clips it. Predict how the simulator’s resource constraints will reduce requests, then choose which shipments to trim first using their existing heuristic priority and physically feasible delivery dates. Where possible, move displaced quantity to a feasible alternative slot, while rechecking source stock, edge capacity, fleet, permissions, and receiver storage.

Before implementing:
1. Freeze the exact v17 and v15 baseline files and parameters.
2. Capture requested flows, predicted clipped flows, actual executed flows, and resulting shortage/cost effects on replayed episodes.
3. Confirm the simulator’s clipping order and limits from project code or authoritative package behavior; do not rely on guessed rules.
4. If runtime access is missing, record the blocker and keep the implementation marked experimental.

Implementation:
- Add the repair as an opt-in layer after the current heuristic stages.
- Make its projection match simulator behavior for edge, source, and fleet limits, including any release reservations that affect available fleet.
- Do not create flow above observed feasibility, use hidden information, or count a shipment twice.
- Leave actions unchanged when no relevant limit binds.
- Add focused checks for competing slots sharing each resource, alternate routes, unknown observations, and receiver overflow.

Evaluation:
Compare against unchanged v17 on the same scenarios under TO_DO.md’s Full and Small protocol. Track RSS by harm level, paired intervals, clipping frequency and quantity, displaced shipments, shortage cost, total shipping/holding cost, fallbacks, and CPU. Include root 0 only for final confirmation. Do not call clipping reduction an RSS gain unless the paired score and causal metrics support it.
```