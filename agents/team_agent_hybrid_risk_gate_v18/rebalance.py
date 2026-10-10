"""Redistribute a source's scarce stock without reducing its dispatch budget."""

import math

import numpy as np


def waterfill(caps, priorities, budget):
    caps = np.asarray(caps, dtype=float)
    priorities = np.asarray(priorities, dtype=float)
    result = np.zeros_like(caps)
    free = caps > 0
    remaining = min(float(budget), float(caps.sum()))
    while remaining > 0 and free.any():
        ids = np.flatnonzero(free)
        weights = caps[ids] * priorities[ids]
        total = float(weights.sum())
        if total == 0:
            weights, total = caps[ids], float(caps[ids].sum())
        proposal = remaining * weights / total
        bound = proposal >= caps[ids]
        if not bound.any():
            result[ids] = proposal
            break
        saturated = ids[bound]
        result[saturated] = caps[saturated]
        remaining = max(0.0, remaining - float(caps[saturated].sum()))
        free[saturated] = False
    return result


class StockRebalancer:
    def __init__(self, config, network, *, fuel_power=0.0, sink_power=0.0, cover_floor=1.0,
                 stock_first=False, fill_spare=False, pipeline_horizon=0, rate_power=0.0, production_power=0.0,
                 fuel_industry_bonus=0.0, priority_first=False, production_value_power=0.0,
                 production_energy_power=0.0, fuel_mark_rate_floor=0.0, fuel_margin_bonus=0.0,
                 production_output_power=0.0, production_wip_horizon=0):
        for name, value in (("fuel_power", fuel_power), ("sink_power", sink_power),
                            ("cover_floor", cover_floor), ("rate_power", rate_power),
                            ("production_power", production_power), ("fuel_industry_bonus", fuel_industry_bonus),
                            ("production_value_power", production_value_power),
                            ("production_energy_power", production_energy_power),
                            ("fuel_mark_rate_floor", fuel_mark_rate_floor), ("fuel_margin_bonus", fuel_margin_bonus),
                            ("production_output_power", production_output_power)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (name == "cover_floor" and value == 0)):
                sign = "positive" if name == "cover_floor" else "nonnegative"
                raise ValueError(f"{name} must be a finite {sign} number")
        if fuel_mark_rate_floor > 1:
            raise ValueError("fuel_mark_rate_floor must be in [0, 1]")
        if not isinstance(stock_first, bool):
            raise ValueError("stock_first must be a boolean")
        if not isinstance(priority_first, bool):
            raise ValueError("priority_first must be a boolean")
        if not isinstance(fill_spare, bool):
            raise ValueError("fill_spare must be a boolean")
        if isinstance(pipeline_horizon, bool) or not isinstance(pipeline_horizon, int) or pipeline_horizon < 0:
            raise ValueError("pipeline_horizon must be a nonnegative integer")
        if (isinstance(production_wip_horizon, bool) or not isinstance(production_wip_horizon, int)
                or production_wip_horizon < 0):
            raise ValueError("production_wip_horizon must be a nonnegative integer")
        self.enabled = bool(fuel_power or sink_power or stock_first or fill_spare or production_power
                            or production_value_power or production_energy_power or production_output_power)
        self.stock_first = stock_first
        self.priority_first = priority_first
        self.fill_spare = fill_spare
        self.pipeline_horizon = pipeline_horizon
        self.cached_observation, self.incoming = None, {}
        self.fuel_power, self.sink_power, self.cover_floor = fuel_power, sink_power, cover_floor
        self.rate_power = rate_power
        self.production_power = production_power
        self.production_output_power = production_output_power
        self.production_wip_horizon = production_wip_horizon
        self.production_value_power, self.production_energy_power = production_value_power, production_energy_power
        self.fuel_industry_bonus = fuel_industry_bonus
        self.fuel_mark_rate_floor = fuel_mark_rate_floor
        self.fuel_margin_bonus = fuel_margin_bonus
        self.network = network
        if not self.enabled:
            return
        names = {name: i for i, name in enumerate(network.node_names)}
        goods = {name: i for i, name in enumerate(network.commodity_names)}
        profiles = {names[node["id"]]: node for node in config["static"]["instance"]["nodes"]}
        self.stocks = {tuple(pair): row for row, pair in enumerate(config["layout"]["stock_slots"])}
        self.demands = {tuple(pair): row for row, pair in enumerate(config["layout"]["demands"])}
        self.nominal_demand = {
            pair: float(profiles[pair[0]]["sink"]["demand"][network.commodity_names[pair[1]]]["dbar"])
            for pair in self.demands
        }
        self.entry_slots = tuple(
            tuple(slot for slot in slots if network.routes[slot].edge_id == edge)
            for edge, slots in enumerate(network.edges_to_slots)
        )
        self.source_slots = tuple((pair, tuple(slots)) for pair, slots in network.slots_from.items())
        self.values = np.ones(len(network.routes))
        for route in network.routes:
            pair = route.destination_node, route.commodity_id
            if pair in self.demands:
                self.values[route.slot_id] = float(
                    profiles[pair[0]]["sink"]["demand"][network.commodity_names[pair[1]]].get("pi", 1)
                )
        self.fuel_rates = {}
        self.production_rates = {}
        self.production_outputs = {}
        if production_output_power:
            for node, attrs in profiles.items():
                fab, osat = attrs.get("fab"), attrs.get("osat")
                products = ({fab["input"]: fab["product"]} if fab else osat["packages"] if osat else {})
                for raw, packed in products.items():
                    storage = attrs.get("stock", {}).get(packed, {}).get("storage")
                    if storage is not None and float(storage) > 0:
                        self.production_outputs[node, goods[raw]] = (node, goods[packed]), float(storage)
        self.production_values, self.fab_grids, self.grid_members = {}, {}, {}
        self.grid_rows = {node: row for row, node in enumerate(config["layout"].get("grids", ()))}
        self.grid_attrs = {node: profiles[node]["grid"] for node in self.grid_rows}
        if production_value_power or production_energy_power:
            penalties = {}
            for profile in profiles.values():
                for name, attrs in profile.get("sink", {}).get("demand", {}).items():
                    penalties[name] = max(penalties.get(name, 0.0), float(attrs["pi"]))
            for node in config["layout"].get("fabs", ()):
                fab = profiles[node]["fab"]
                self.production_values[node, goods[fab["input"]]] = penalties.get(
                    fab["product"].removesuffix("_raw"), 0.0)
                if fab["grid"] is not None:
                    self.fab_grids[node, goods[fab["input"]]] = names[fab["grid"]]
            for node in config["layout"].get("osats", ()):
                for raw, product in profiles[node]["osat"]["packages"].items():
                    self.production_values[node, goods[raw]] = penalties.get(product, 0.0)
            maximum = max(self.production_values.values(), default=1.0)
            self.production_values = {pair: value / maximum if maximum else 1.0
                                      for pair, value in self.production_values.items()}
        for row, node in enumerate(config["layout"].get("fabs", ())):
            fab = profiles[node]["fab"]
            self.production_rates[node, goods[fab["input"]]] = ("fab.cap_eff", row, float(fab["cap0"]), 1.0)
        for row, node in enumerate(config["layout"].get("osats", ())):
            osat = profiles[node]["osat"]
            total = sum(float(profiles[node]["stock"][raw]["storage"]) for raw in osat["packages"])
            for raw in osat["packages"]:
                fraction = float(profiles[node]["stock"][raw]["storage"]) / total if total else 0.0
                self.production_rates[node, goods[raw]] = ("osat.thr_eff", row, float(osat["thr"]), fraction)
        self.fuel_consumers = {}
        self.industry = {}
        for row, node in enumerate(config["layout"].get("fabs", ())):
            fab = profiles[node]["fab"]
            if fab["grid"] is None:
                continue
            grid_node = names[fab["grid"]]
            if production_energy_power:
                self.grid_members.setdefault(grid_node, []).append((row, float(fab["cap0"]), float(fab["e"])))
            output = fab["product"].removesuffix("_raw")
            penalties = [float(profile["sink"]["demand"][output]["pi"])
                         for profile in profiles.values()
                         if output in profile.get("sink", {}).get("demand", {})]
            self.industry.setdefault(grid_node, []).append((row, float(fab["cap0"]), float(fab["e"]),
                                                            max(penalties, default=0.0)))
        self.voll = {node: float(profiles[node]["grid"]["voll"]) for node in self.industry}
        self.goods, self.psi = goods, float(config["static"]["instance"].get("params", {}).get("psi", 0))
        for node in config["layout"]["grids"]:
            grid = profiles[node]["grid"]
            for name, share in grid["shares"].items():
                if name in goods and share > 0:
                    pair = node, goods[name]
                    self.fuel_rates[pair] = float(grid["deliverable"]) * float(share)
                    self.fuel_consumers[pair] = (pair,)
        grids = set(config["layout"]["grids"])
        connected = {}
        for route in network.routes:
            pair = route.destination_node, route.commodity_id
            if route.destination_node in grids and pair in self.fuel_rates:
                connected.setdefault((route.source_node, route.commodity_id), set()).add(pair)
        for pair, consumers in connected.items():
            if pair in self.fuel_rates:
                continue
            self.fuel_rates[pair] = sum(self.fuel_rates[child] for child in consumers)
            self.fuel_consumers[pair] = (pair, *sorted(consumers))

    def _stock(self, pair, observation):
        row = self.stocks.get(pair)
        if row is None or not observation["stock.qty.observed"][row]:
            return None
        return float(observation["stock.qty"][row])

    def _coverage(self, route, observation):
        return self.coverage((route.destination_node, route.commodity_id), observation)

    def coverage(self, pair, observation):
        self._incoming(observation)
        if self.fuel_power and pair in self.fuel_rates:
            quantities = [self._stock(child, observation) for child in self.fuel_consumers[pair]]
            if any(q is None for q in quantities):
                return None, self.fuel_power
            arrived = sum(self.incoming.get(child, 0.0) for child in self.fuel_consumers[pair])
            return (sum(quantities) + arrived) / self.fuel_rate(pair, observation), self.fuel_power
        if self.sink_power and pair in self.demands:
            row = self.demands[pair]
            seen = observation["demand_forecast.qty.observed"][row].astype(bool)
            rate = (float(observation["demand_forecast.qty"][row][seen].mean()) if seen.any()
                    else self.nominal_demand[pair])
            stock = self._stock(pair, observation)
            if stock is None or rate <= 0:
                return None, self.sink_power
            return (stock + self.incoming.get(pair, 0.0)) / rate, self.sink_power
        if self.production_power and pair in self.production_rates:
            rate = self.production_rate(pair, observation)
            stock = self._stock(pair, observation)
            if stock is None or rate <= 0:
                return None, self.production_power
            return (stock + self.incoming.get(pair, 0.0)) / rate, self.production_power
        return None, 0.0

    def production_rate(self, pair, observation):
        key, row, nominal, fraction = self.production_rates[pair]
        value = (float(observation[f"graph_now.{key}"][row])
                 if observation[f"graph_now.{key}.observed"][row] else nominal)
        return value * fraction

    def fuel_rate(self, pair, observation):
        if not self.fuel_mark_rate_floor:
            return self.fuel_rates[pair]
        total = 0.0
        for child in self.fuel_consumers[pair]:
            node, _commodity = child
            if node not in self.grid_rows:
                continue
            row = self.grid_rows[node]
            nominal = float(self.grid_attrs[node]["deliverable"])
            key = "graph_now.grid.G_bar"
            seen = key in observation and observation[key + ".observed"][row]
            ratio = float(observation[key][row]) / nominal if seen and nominal > 0 else 1.0
            total += self.fuel_rates[child] * max(self.fuel_mark_rate_floor, min(1.0, ratio))
        return total if total > 0 else self.fuel_rates[pair]

    def fuel_priority(self, pair, observation):
        if (not self.fuel_industry_bonus and not self.fuel_margin_bonus) or pair not in self.fuel_consumers:
            return 1.0
        premium = 0.0
        for node, _k in self.fuel_consumers[pair]:
            profit, energy = 0.0, 0.0
            for row, cap, draw, penalty in self.industry.get(node, ()):
                r_key, a_key = "graph_now.fab.R", "graph_now.fab.alpha_bar"
                restoration = float(observation[r_key][row]) if observation[r_key + ".observed"][row] else 1.0
                alpha = float(observation[a_key][row]) if observation[a_key + ".observed"][row] else 1.0
                profit += restoration * alpha * cap * penalty
                energy += alpha * cap * draw if restoration > 0 else 0.0
            if energy > 0 and self.voll.get(node, 0) > 0:
                premium = max(premium, profit / energy / self.voll[node])
        margin = max((self._fuel_margin(node, observation) for node, _k in self.fuel_consumers[pair]), default=0.0)
        return 1.0 + self.fuel_industry_bonus * premium + self.fuel_margin_bonus * margin

    def _fuel_margin(self, node, observation):
        if not self.fuel_margin_bonus or node not in self.grid_rows or not self.industry.get(node):
            return 0.0
        row, grid = self.grid_rows[node], self.grid_attrs[node]
        g_key, y_key = "graph_now.grid.G_bar", "graph_now.grid.y_bar"
        if not (observation[g_key + ".observed"][row] and observation[y_key + ".observed"][row]):
            return 0.0
        generation, load = float(observation[g_key][row]), float(observation[y_key][row])
        if generation <= 0 or self.voll[node] <= 0:
            return 0.0
        available = 0.0
        for name, share in grid["shares"].items():
            segment = float(share) * generation
            if name in self.goods and name not in ("lng", "crude"):
                stock = self._stock((node, self.goods[name]), observation)
                if stock is None:
                    return 0.0
                segment = min(segment, stock + self.incoming.get((node, self.goods[name]), 0.0))
            available += segment
        profit, draw = 0.0, 0.0
        for fab_row, nominal, energy, penalty in self.industry[node]:
            r_key, a_key, c_key = "graph_now.fab.R", "graph_now.fab.alpha_bar", "graph_now.fab.cap_eff"
            r = float(observation[r_key][fab_row]) if observation[r_key + ".observed"][fab_row] else 1.0
            if r <= 0:
                continue
            alpha = float(observation[a_key][fab_row]) if observation[a_key + ".observed"][fab_row] else 1.0
            capacity = (float(observation[c_key][fab_row]) if observation[c_key + ".observed"][fab_row]
                        else nominal * r * alpha)
            profit += capacity * penalty
            draw += energy * capacity / r
        if draw <= 0:
            return 0.0
        priority = grid.get("priority", "base_first")
        fraction = min(1.0, max(0.0, available - load) / draw) if priority == "base_first" else min(
            1.0, available / (draw + load if priority == "proportional" else draw))
        # Fully fuelled, persistent-current-mark proxy; not an arrival-time promise.
        return min(3.0, fraction * profit / generation / self.voll[node])

    def production_priority(self, pair, observation):
        value = max(0.01, self.production_values.get(pair, 1.0)) ** self.production_value_power
        node = self.fab_grids.get(pair)
        if not self.production_energy_power or node is None:
            return value
        row, grid = self.grid_rows[node], self.grid_attrs[node]
        generation_key, load_key = "graph_now.grid.G_bar", "graph_now.grid.y_bar"
        if not (observation[generation_key + ".observed"][row] and observation[load_key + ".observed"][row]):
            return value
        generation, available = float(observation[generation_key][row]), 0.0
        for name, share in grid["shares"].items():
            segment = float(share) * generation
            if name in self.goods:
                stock = self._stock((node, self.goods[name]), observation)
                if stock is None:
                    return value
                if name == grid.get("rationed"):
                    threshold = self.psi * float(grid["ibar"].get(name, 0))
                    segment *= min(1.0, stock / threshold) if threshold > 0 else 1.0
                segment = min(segment, stock)
            available += segment
        draw = 0.0
        for fab_row, nominal, energy in self.grid_members.get(node, ()):
            key = "graph_now.fab.R"
            restoration = float(observation[key][fab_row]) if observation[key + ".observed"][fab_row] else 1.0
            if restoration <= 0:
                continue
            key = "graph_now.fab.cap_eff"
            capacity = (float(observation[key][fab_row]) if observation[key + ".observed"][fab_row]
                        else nominal * restoration)
            draw += energy * capacity / restoration
        load = float(observation[load_key][row])
        if grid.get("priority", "base_first") == "base_first":
            fraction = min(1.0, max(0.0, available - load) / draw) if draw > 0 else 1.0
        elif grid["priority"] == "industrial_first":
            fraction = min(1.0, available / draw) if draw > 0 else 1.0
        else:
            fraction = min(1.0, available / (load + draw)) if load + draw > 0 else 1.0
        # Current energy is only a proxy for arrival-time readiness. Keep a floor.
        return value * (0.25 + 0.75 * fraction) ** self.production_energy_power

    def _incoming(self, observation):
        if not self.pipeline_horizon or self.cached_observation is observation:
            return
        self.cached_observation, self.incoming = observation, {}
        week = int(observation["week"][0])
        lane_edges = set(edge for path in self.network.lane_edges for edge in path)
        for row in np.flatnonzero(observation["pipeline.qty.observed"]):
            if not all(observation[f"pipeline.{name}.observed"][row] for name in ("edge", "k", "arrival_week")):
                continue
            edge = int(observation["pipeline.edge"][row])
            if observation["pipeline.lane.observed"][row]:
                lane = int(observation["pipeline.lane"][row])
            elif edge not in lane_edges:
                lane = None
            else:
                continue
            if lane is None:
                destination, remaining = self.network.edge_head[edge], ()
                if destination in self.network.chokepoints:
                    continue
            else:
                progress = self.network.transit_progress(edge, lane)
                if progress.remaining_chokepoints:
                    continue
                destination, remaining = progress.destination_node, progress.remaining_edges
                if not all(observation["graph_now.tau.observed"][e] for e in remaining):
                    continue
            due = int(observation["pipeline.arrival_week"][row])
            due += sum(int(observation["graph_now.tau"][e]) for e in remaining)
            if week <= due <= week + self.pipeline_horizon:
                pair = destination, int(observation["pipeline.k"][row])
                self.incoming[pair] = self.incoming.get(pair, 0.0) + float(observation["pipeline.qty"][row])

    def apply(self, flows, observation):
        if not self.enabled:
            return flows
        caps = flows.copy()
        if self.priority_first:
            for pair, slots in self.network.slots_from.items():
                slots = list(slots)
                stock = self._stock(pair, observation)
                if stock is not None and stock < float(caps[slots].sum()):
                    priorities, active = self._priorities(slots, caps, observation)
                    if active:
                        caps[slots] = waterfill(caps[slots], priorities, stock)
        if self.stock_first:
            for pair, slots in self.network.slots_from.items():
                slots = list(slots)
                stock = self._stock(pair, observation)
                total = float(caps[slots].sum())
                if stock is not None and total > stock:
                    caps[slots] *= stock / total
        for edge, slots in enumerate(self.entry_slots):
            # Only an entry edge has a request; a route's later edges do not.
            slots = [slot for slot in slots if self.network.routes[slot].edge_id == edge]
            if slots and observation["graph_now.u.observed"][edge]:
                total = float(caps[slots].sum())
                if total:
                    caps[slots] *= min(1.0, float(observation["graph_now.u"][edge]) / total)
        result = caps.copy()
        for pair, slots in self.network.slots_from.items():
            slots = list(slots)
            stock = self._stock(pair, observation)
            total = float(caps[slots].sum())
            if stock is None or total == 0:
                continue
            budget = min(stock, total)
            if budget >= total:
                continue
            priorities, active = self._priorities(slots, caps, observation)
            if active:
                result[slots] = waterfill(caps[slots], priorities, budget)
            else:
                result[slots] *= budget / total
        if self.fill_spare:
            result = self._fill_spare(result, flows, observation)
        return result

    def _output_wip(self, observation):
        if not self.production_output_power or not self.production_wip_horizon:
            return {}
        keys = tuple(f"wip.{field}" for field in ("qty", "node", "k", "out_week"))
        if not all(key in observation and key + ".observed" in observation for key in keys):
            return {}
        seen = np.logical_and.reduce([observation[key + ".observed"].astype(bool) for key in keys])
        week = int(observation["week"][0])
        totals = {}
        for row in np.flatnonzero(seen):
            due = int(observation["wip.out_week"][row])
            if week <= due <= week + self.production_wip_horizon:
                pair = int(observation["wip.node"][row]), int(observation["wip.k"][row])
                # Gross visible WIP is a congestion proxy; future scrap and exports remain unknown.
                totals[pair] = totals.get(pair, 0.0) + max(0.0, float(observation["wip.qty"][row]))
        return totals

    def _priorities(self, slots, caps, observation):
        priorities, missing, active = [], [], False
        wip = self._output_wip(observation)
        for slot in slots:
            route = self.network.routes[slot]
            coverage, power = self._coverage(route, observation)
            active |= bool(power)
            missing.append(coverage is None)
            priorities.append((self.cover_floor / max(self.cover_floor, coverage)) ** power
                              if coverage is not None and power else 1.0)
            priorities[-1] *= self.fuel_priority((route.destination_node, route.commodity_id), observation)
            if self.production_output_power:
                pair = route.destination_node, route.commodity_id
                if pair in self.production_outputs:
                    output, storage = self.production_outputs[pair]
                    stock = self._stock(output, observation)
                    if stock is not None:
                        # Current output congestion is a soft priority, not a future export prediction.
                        stock += wip.get(output, 0.0)
                        spare = max(0.05, 1.0 - min(1.0, max(0.0, stock / storage)))
                        priorities[-1] *= spare ** self.production_output_power
                        if not power:
                            missing[-1] = False
                        active = True
            if self.production_value_power or self.production_energy_power:
                pair = route.destination_node, route.commodity_id
                priorities[-1] *= self.production_priority(pair, observation)
                active |= pair in self.production_values
        if active:
            largest = max((p for p, hidden in zip(priorities, missing) if not hidden), default=1.0)
            priorities = [largest if hidden else p for p, hidden in zip(priorities, missing)]
            if self.rate_power:
                totals = {}
                for slot in slots:
                    route = self.network.routes[slot]
                    target = route.destination_node, route.commodity_id
                    totals[target] = totals.get(target, 0.0) + caps[slot]
                for i, slot in enumerate(slots):
                    route = self.network.routes[slot]
                    target = route.destination_node, route.commodity_id
                    rate = self.fuel_rates.get(target, self.nominal_demand.get(target))
                    if target in self.fuel_rates:
                        rate = self.fuel_rate(target, observation)
                    if target in self.production_rates:
                        rate = self.production_rate(target, observation)
                    if rate is not None and totals[target] > 0:
                        priorities[i] *= (rate / totals[target]) ** self.rate_power
        utility = getattr(self, "raw_utility", None)
        if active and utility is not None:
            priorities = utility.adjust(priorities, slots, caps, observation)
        return priorities, active

    def _fill_spare(self, result, requests, observation):
        from scipy.optimize import linprog

        upper = np.maximum(0.0, requests - result)
        rows, limits = [], []
        for pair, slots in self.source_slots:
            stock = self._stock(pair, observation)
            ids = list(slots)
            if stock is None:
                upper[ids] = 0
                continue
            row = np.zeros(len(result))
            row[ids] = 1
            rows.append(row)
            limits.append(max(0.0, stock - float(result[ids].sum())))
        for edge, slots in enumerate(self.entry_slots):
            if not slots:
                continue
            ids = list(slots)
            if not observation["graph_now.u.observed"][edge]:
                upper[ids] = 0
                continue
            row = np.zeros(len(result))
            row[ids] = 1
            rows.append(row)
            limits.append(max(0.0, float(observation["graph_now.u"][edge]) - float(result[ids].sum())))
        if not upper.any():
            return result
        # Additive optimization cannot take away the baseline's feasible cargo.
        solution = linprog(-self.values / self.values.max(), A_ub=np.asarray(rows), b_ub=limits,
                           bounds=np.column_stack((np.zeros(len(result)), upper)), method="highs")
        if not solution.success:
            raise RuntimeError(f"spare-capacity optimization failed: {solution.message}")
        return result + np.minimum(upper, np.maximum(0.0, solution.x))
