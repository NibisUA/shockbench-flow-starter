"""Opt-in tanker redirection using visible queue content and receiver stocks."""

import math
from types import SimpleNamespace

import numpy as np
from queue_forecast import QueueForecaster
from queue_delay import QueueDelay
from queue_outedge import OutEdgeQueueDelay
from rebalance import StockRebalancer, waterfill


class FuelRelease:
    def __init__(self, config, network, *, power=0.0, transit_bias=0.0, pipeline_horizon=0,
                 rate_power=0.0, cover_floor=1.0, only_blocked=False, preserve_routes=False,
                 resource_safe=False, queue_first_fleet=False, industry_bonus=0.0, receiver_cover=0.0,
                 origin_bonus=1.25, fuel_mark_rate_floor=0.0, fuel_margin_bonus=0.0, origin_urgency=0.0, queue_delay=False, queue_discount=0.05, outedge_delay=False):
        for name, value in (("power", power), ("transit_bias", transit_bias), ("rate_power", rate_power),
                            ("receiver_cover", receiver_cover), ("origin_bonus", origin_bonus),
                            ("origin_urgency", origin_urgency), ("queue_discount", queue_discount)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(outedge_delay, bool):
            raise ValueError("outedge_delay must be a boolean")
        if outedge_delay and not queue_delay:
            raise ValueError("outedge_delay requires queue_delay")
        delay_type = OutEdgeQueueDelay if outedge_delay else QueueDelay
        self.queue_delay = delay_type(config, network, enabled=queue_delay)
        self.queue_discount = queue_discount
        self.enabled = bool(power)
        if not isinstance(only_blocked, bool) or not isinstance(preserve_routes, bool):
            raise ValueError("only_blocked and preserve_routes must be booleans")
        self.only_blocked, self.preserve_routes = only_blocked, preserve_routes
        if not isinstance(resource_safe, bool) or not isinstance(queue_first_fleet, bool):
            raise ValueError("resource_safe and queue_first_fleet must be booleans")
        self.resource_safe, self.queue_first_fleet = resource_safe, queue_first_fleet
        self.receiver_cover = receiver_cover
        self.origin_bonus = origin_bonus
        self.origin_urgency = origin_urgency
        if not self.enabled:
            return
        self.power, self.transit_bias = power, transit_bias
        self.rate_power = rate_power
        self.network = network
        self.receiver_nodes = {
            i for i, kind in enumerate(config["static"]["nodes"]["type"])
            if kind in ("terminal", "grid")
        }
        self.cover = StockRebalancer(config, network, fuel_power=power, pipeline_horizon=pipeline_horizon,
                                    cover_floor=cover_floor, fuel_industry_bonus=industry_bonus,
                                    fuel_mark_rate_floor=fuel_mark_rate_floor, fuel_margin_bonus=fuel_margin_bonus)
        if not self.cover.fuel_rates:
            self.enabled = False
            return
        self.cover_floor = cover_floor
        self.chokes = tuple(config["layout"]["chokepoints"])
        self.pairs = tuple(map(tuple, config["layout"]["release_pairs"]))
        keys = config["layout"].get("lot_keys")
        self.lot_keys = None if keys is None else tuple(map(tuple, keys))
        overrides = config["static"]["override_slots"]
        self.slots = tuple(zip(overrides["chokepoint"], overrides["k"],
                               overrides["out_edge"], overrides["lane"], strict=True))
        self.paths, self.destinations = [], []
        self.original_slots = {tuple(item): slot for slot, item in enumerate(self.slots)}
        if queue_first_fleet:
            fleet = QueueForecaster(config)
            self.fleet_caps = fleet.fleet_caps
            self.flow_weights = np.asarray([
                sum(delta for lane, delta in fleet.fleet_terms.get(r.edge_id, ())
                    if lane is None or lane == r.lane_id) if r.pool == "tb" else 0.0 for r in network.routes
            ])
            self.release_weights = np.asarray([
                sum(delta for match, delta in fleet.fleet_terms.get(edge, ()) if match is None or match == lane)
                for _node, _k, edge, lane in self.slots
            ])
        for _c, _k, edge, lane in self.slots:
            path = ((edge,) if lane is None else
                    network.lane_edges[lane][network.lane_edges[lane].index(edge):])
            self.paths.append(path)
            self.destinations.append(network.edge_head[path[-1]])
        self.queue_routes = tuple(SimpleNamespace(edges=path, commodity_id=item[1])
                                  for path, item in zip(self.paths, self.slots, strict=True))

    def apply(self, action, observation):
        if not self.enabled:
            return action
        # In a blackout, retain the default FIFO rather than trust padded rows.
        if not observation["override_mask.observed"][0]:
            return action
        week = int(observation["week"][0])
        if self.queue_delay.enabled:
            self.queue_delay.begin(observation)
        forecast = getattr(self, "warning_forecast", None)
        risks = {} if forecast is None else forecast.probabilities(observation)
        content = {}
        original, blocked = np.zeros(len(self.slots)), set()
        quantities = observation["queue_lots.qty"]
        seen = observation["queue_lots.qty.observed"].astype(bool)

        def track(choke, commodity, lane, edge, amount):
            slot = self.original_slots.get((choke, commodity, edge, lane))
            if slot is not None and amount > 0:
                original[slot] += amount
                if not self._passable(slot, observation):
                    blocked.add((choke, commodity))

        if self.lot_keys is None:
            for row in np.flatnonzero(seen):
                if not all(observation[f"queue_lots.{field}.observed"][row] for field in ("chokepoint", "k")):
                    continue
                choke, commodity = (int(observation[f"queue_lots.{field}"][row]) for field in ("chokepoint", "k"))
                amount = float(quantities[row])
                content[choke, commodity] = content.get((choke, commodity), 0.0) + amount
                if all(observation[f"queue_lots.{field}.observed"][row] for field in ("lane", "next_edge")):
                    lane = int(observation["queue_lots.lane"][row])
                    track(choke, commodity, None if lane < 0 else lane,
                          int(observation["queue_lots.next_edge"][row]), amount)
        else:
            for row, (choke, commodity, lane, edge) in enumerate(self.lot_keys):
                amount = float(quantities[row][seen[row]].sum())
                content[choke, commodity] = content.get((choke, commodity), 0.0) + amount
                track(choke, commodity, lane, edge, amount)
        # These lots enter the queue before this week's release, not afterwards.
        for row in np.flatnonzero(observation["pipeline.qty.observed"]):
            if not all(observation[f"pipeline.{field}.observed"][row]
                       for field in ("edge", "k", "arrival_week")):
                continue
            if int(observation["pipeline.arrival_week"][row]) != week:
                continue
            node = self.network.edge_head[int(observation["pipeline.edge"][row])]
            pair = node, int(observation["pipeline.k"][row])
            if node in self.chokes:
                amount = float(observation["pipeline.qty"][row])
                content[pair] = content.get(pair, 0.0) + amount
                if observation["pipeline.lane.observed"][row]:
                    lane = int(observation["pipeline.lane"][row])
                    path = self.network.lane_edges[lane]
                    edge = int(observation["pipeline.edge"][row])
                    track(*pair, lane, path[path.index(edge) + 1], amount)
        qty = np.zeros(len(self.slots))
        limits, scores = np.zeros(len(self.slots)), np.zeros(len(self.slots))
        modes = np.zeros(len(self.pairs), dtype=np.int64)
        for pos, pair in enumerate(self.pairs):
            choke, commodity = pair
            if self.only_blocked and pair not in blocked:
                continue
            ci = self.chokes.index(choke)
            budget = content.get(pair, 0.0)
            if not budget or not observation["graph_now.kappa.tb.observed"][ci]:
                continue
            budget = min(budget, float(observation["graph_now.kappa.tb"][ci]))
            slots, caps, priorities = [], [], []
            for slot, (c, k, edge, _lane) in enumerate(self.slots):
                if (c, k) != pair or not observation["override_mask"][slot]:
                    continue
                destination = self.destinations[slot], commodity
                if destination[0] not in self.receiver_nodes:
                    continue
                if destination not in self.cover.fuel_rates or not observation["graph_now.u.observed"][edge]:
                    continue
                path = self.paths[slot]
                if any(observation["graph_now.prohibited.observed"][e, k]
                       and observation["graph_now.prohibited"][e, k] for e in path):
                    continue
                if self.only_blocked and not self._passable(slot, observation):
                    continue
                factor = 1.0
                for e in path[1:]:
                    node = self.network.edge_tail[e]
                    if node in self.chokes:
                        cp = self.chokes.index(node)
                        if observation["graph_now.open.observed"][cp]:
                            factor *= float(observation["graph_now.open"][cp])
                coverage, _ = self.cover.coverage(destination, observation)
                coverage = 1.0 if coverage is None else max(self.cover_floor, coverage)
                distance = sum(float(observation["graph_now.tau"][e])
                               if observation["graph_now.tau.observed"][e]
                               else self.network.edge_transit_weeks[e] for e in path)
                slots.append(slot)
                caps.append(float(observation["graph_now.u"][edge]) * factor)
                priorities.append((self.cover_floor / coverage) ** self.power
                                  * math.exp(-min(700, self.transit_bias * distance)))
                priorities[-1] *= self.cover.fuel_priority(destination, observation)
                if self.queue_delay.enabled:
                    diagnostic = self.queue_delay.route_delay(self.queue_routes[slot], observation, distance)
                    wait = diagnostic["wait_weeks"]
                    if wait is not None and wait > 0:
                        priorities[-1] *= math.exp(-min(700.0, self.queue_discount * wait))
                if risks:
                    departure, risk = 0.0, 0.0
                    for pos, edge in enumerate(path):
                        if pos and self.network.edge_tail[edge] in risks:
                            probabilities = risks[self.network.edge_tail[edge]]
                            horizon = min(len(probabilities), max(1, math.ceil(departure))) - 1
                            risk += float(probabilities[horizon])
                        departure += (float(observation["graph_now.tau"][edge])
                                      if observation["graph_now.tau.observed"][edge]
                                      else self.network.edge_transit_weeks[edge])
                    priorities[-1] *= math.exp(-min(700.0, self.warning_strength * risk))
            if not slots or not sum(caps) or not sum(priorities):
                continue
            if self.rate_power:
                totals = {}
                for slot, cap in zip(slots, caps):
                    target = self.destinations[slot], commodity
                    totals[target] = totals.get(target, 0.0) + cap
                for i, slot in enumerate(slots):
                    target = self.destinations[slot], commodity
                    if totals[target] > 0:
                        priorities[i] *= (self.cover.fuel_rate(target, observation) / totals[target]) ** self.rate_power
            limits[slots], scores[slots] = caps, priorities
            if self.preserve_routes:
                prior = np.minimum(caps, original[slots])
                prior = np.asarray([amount if self._passable(slot, observation) else 0.0
                                    for slot, amount in zip(slots, prior)])
                if prior.sum() > budget:
                    prior *= budget / prior.sum()
                remaining = np.maximum(0.0, np.asarray(caps) - prior)
                qty[slots] = prior + waterfill(remaining, priorities, max(0.0, budget - float(prior.sum())))
            else:
                qty[slots] = waterfill(caps, priorities, budget)
            if qty[slots].sum() > 0:
                modes[pos] = 1
        if self.resource_safe:
            qty = self._augment(original, limits, scores, content, observation)
        if self.receiver_cover:
            qty = self._receiver_optimize(original, limits, scores, content, observation)
            for pos, pair in enumerate(self.pairs):
                slots = [s for s, item in enumerate(self.slots) if item[:2] == pair]
                if content.get(pair, 0) and any(limits[s] > 0 for s in slots):
                    modes[pos] = 1
        # Out edges and tanker throughput are shared across all override slots.
        for edge in set(slot[2] for slot in self.slots):
            slots = [s for s, item in enumerate(self.slots) if item[2] == edge]
            total = float(qty[slots].sum())
            if total and observation["graph_now.u.observed"][edge]:
                qty[slots] *= min(1.0, float(observation["graph_now.u"][edge]) / total)
        for ci, choke in enumerate(self.chokes):
            slots = [s for s, item in enumerate(self.slots) if item[0] == choke]
            total = float(qty[slots].sum())
            if total and observation["graph_now.kappa.tb.observed"][ci]:
                qty[slots] *= min(1.0, float(observation["graph_now.kappa.tb"][ci]) / total)
        if self.queue_first_fleet:
            action = self._fleet_first(action, qty)
        return {**action, "override_qty": qty, "release_mode": modes}

    def _receiver_optimize(self, original, limits, scores, content, observation):
        from scipy.optimize import linprog

        size = len(limits)
        prior = np.minimum(original, limits)
        upper = np.concatenate((prior, np.maximum(0, limits - prior)))
        if not upper.any():
            return np.zeros(size)
        rows, rhs = [], []

        def bound(slots, cap):
            row = np.zeros(2 * size)
            row[slots] = 1
            row[np.asarray(slots, dtype=int) + size] = 1
            rows.append(row)
            rhs.append(max(0.0, cap))

        for pair in self.pairs:
            bound([s for s, item in enumerate(self.slots) if item[:2] == pair], content.get(pair, 0))
        for edge in set(item[2] for item in self.slots):
            if observation["graph_now.u.observed"][edge]:
                bound([s for s, item in enumerate(self.slots) if item[2] == edge], observation["graph_now.u"][edge])
        for ci, choke in enumerate(self.chokes):
            if observation["graph_now.kappa.tb.observed"][ci]:
                bound([s for s, item in enumerate(self.slots) if item[0] == choke],
                      observation["graph_now.kappa.tb"][ci])
        receivers = {}
        for s, item in enumerate(self.slots):
            if limits[s] > 0:
                receivers.setdefault((self.destinations[s], item[1]), []).append(s)
        for pair, slots in receivers.items():
            coverage, _power = self.cover.coverage(pair, observation)
            if coverage is None:
                continue
            distance = {s: sum(float(observation["graph_now.tau"][e])
                               if observation["graph_now.tau.observed"][e]
                               else self.network.edge_transit_weeks[e] for e in self.paths[s]) for s in slots}
            for delay in sorted(set(distance.values())):
                cap = self.cover.fuel_rate(pair, observation) * (self.receiver_cover + delay - coverage)
                bound([s for s in slots if distance[s] <= delay], cap)
        # A mild origin preference retains feasible plans without forcing oversupply.
        origin_weights = np.full(size, self.origin_bonus)
        if self.origin_urgency:
            for s in range(size):
                pair = self.destinations[s], self.slots[s][1]
                if pair not in self.cover.fuel_rates:
                    continue
                coverage, _power = self.cover.coverage(pair, observation)
                if coverage is not None:
                    origin_weights[s] += self.origin_urgency / max(1.0, coverage)
        objective = np.concatenate((scores * origin_weights, scores))
        solved = linprog(-objective, A_ub=np.asarray(rows), b_ub=rhs,
                         bounds=np.column_stack((np.zeros(2 * size), upper)), method="highs")
        if not solved.success:
            raise RuntimeError(f"receiver queue optimization failed: {solved.message}")
        x = np.minimum(upper, np.maximum(0, solved.x))
        return x[:size] + x[size:]

    def _augment(self, original, limits, scores, content, observation):
        from scipy.optimize import linprog

        base = np.minimum(original, limits)
        for slot in range(len(base)):
            if base[slot] and not self._passable(slot, observation):
                base[slot] = 0
        groups = []
        for edge in set(item[2] for item in self.slots):
            slots = [s for s, item in enumerate(self.slots) if item[2] == edge]
            if observation["graph_now.u.observed"][edge]:
                groups.append((slots, float(observation["graph_now.u"][edge])))
        for ci, choke in enumerate(self.chokes):
            slots = [s for s, item in enumerate(self.slots) if item[0] == choke]
            if observation["graph_now.kappa.tb.observed"][ci]:
                groups.append((slots, float(observation["graph_now.kappa.tb"][ci])))
        for slots, cap in groups:
            total = float(base[slots].sum())
            if total > cap:
                base[slots] *= max(0.0, cap) / total
        rows, rhs = [], []
        for pair in self.pairs:
            slots = [s for s, item in enumerate(self.slots) if item[:2] == pair]
            groups.append((slots, content.get(pair, 0.0)))
        for slots, cap in groups:
            row = np.zeros(len(base))
            row[slots] = 1
            rows.append(row)
            rhs.append(max(0.0, cap - float(base[slots].sum())))
        upper = np.maximum(0.0, limits - base)
        if not upper.any():
            return base
        solved = linprog(-scores, A_ub=np.asarray(rows), b_ub=rhs,
                         bounds=np.column_stack((np.zeros(len(base)), upper)), method="highs")
        if not solved.success:
            raise RuntimeError(f"queue resource optimization failed: {solved.message}")
        return base + np.minimum(upper, np.maximum(0.0, solved.x))

    def _fleet_first(self, action, qty):
        budget = self.fleet_caps["tb"]
        use = float(np.dot(qty, self.release_weights))
        if use > budget:
            qty[self.release_weights > 0] *= budget / use
            use = budget
        flows = action["flows"].copy()
        requested = float(np.dot(flows, self.flow_weights))
        if requested > max(0.0, budget - use):
            flows[self.flow_weights > 0] *= max(0.0, budget - use) / requested
        return {**action, "flows": flows}

    def _passable(self, slot, observation):
        _c, k, _edge, _lane = self.slots[slot]
        if not observation["override_mask"][slot]:
            return False
        for edge in self.paths[slot]:
            if (observation["graph_now.u.observed"][edge] and observation["graph_now.u"][edge] == 0
                    or observation["graph_now.prohibited.observed"][edge, k]
                    and observation["graph_now.prohibited"][edge, k]):
                return False
            node = self.network.edge_tail[edge]
            if node in self.chokes:
                cp = self.chokes.index(node)
                if observation["graph_now.open.observed"][cp] and observation["graph_now.open"][cp] == 0:
                    return False
        return True
