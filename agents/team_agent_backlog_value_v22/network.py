"""V1: static routes and shared resources, using the config's integer indices.

Build StaticNetwork once in Agent(config). This module describes existing
action slots; it does not choose flows, forecast needs or read weekly signals.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from operator import index
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True, slots=True)
class SlotRoute:
    slot_id: int
    commodity_id: int
    source_node: int
    destination_node: int
    edge_id: int
    lane_id: int | None
    edges: tuple[int, ...]
    nodes: tuple[int, ...]
    chokepoints: tuple[int, ...]
    chokepoint_positions: tuple[int, ...]
    pool: str
    unit: str
    nominal_transit_weeks: int
    nominal_freight_per_unit: float


@dataclass(frozen=True, slots=True)
class SharedResources:
    edges: tuple[int, ...]
    chokepoints: tuple[int, ...]
    chokepoint_pools: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class TransitProgress:
    """Where a shipment arrives on its current edge, and what remains afterwards.

    The remaining nominal transit excludes the current edge and queue waiting.
    This describes a route, not a forecast of actual arrival at destination.
    """

    edge_id: int
    lane_id: int | None
    arrival_node: int
    destination_node: int
    remaining_edges: tuple[int, ...]
    remaining_chokepoints: tuple[int, ...]
    remaining_nominal_transit_weeks: int

    @property
    def reaches_destination(self):
        return not self.remaining_edges and self.arrival_node == self.destination_node


def _checked_index(value, size, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer index")
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an integer index") from exc
    if not 0 <= value < size:
        raise ValueError(f"{label} index {value} outside [0, {size})")
    return value


def _freeze_groups(groups):
    return MappingProxyType({key: tuple(values) for key, values in groups.items()})


class StaticNetwork:
    """Topology cached from static tables, without Small/Full slot constants.

    edges_to_slots shares edge capacity across commodities. Chokepoint pools
    share throughput only within tb/ct; a shared physical chokepoint can still
    expose two different pools to the same disruption.
    """

    def __init__(self, config):
        static = config["static"]
        edges, lanes, goods, slots = (static["edges"], static["lanes"], static["commodities"], static["action_slots"])
        self.node_names = tuple(static["nodes"]["id"])
        self.edge_names = tuple(edges["id"])
        self.commodity_names = tuple(goods["id"])
        self.lane_names = tuple(lanes["id"])
        self.edge_tail = tuple(edges["tail"])
        self.edge_head = tuple(edges["head"])
        self.edge_capacity = tuple(edges["u0"])
        self.edge_transit_weeks = tuple(edges["tau0"])
        self.chokepoints = tuple(config["layout"]["chokepoints"])
        positions = {node: p for p, node in enumerate(self.chokepoints)}
        out_edges = [[] for _ in self.node_names]
        in_edges = [[] for _ in self.node_names]
        for e, (tail, head) in enumerate(zip(self.edge_tail, self.edge_head, strict=True)):
            _checked_index(tail, len(self.node_names), "edge tail")
            _checked_index(head, len(self.node_names), "edge head")
            out_edges[tail].append(e)
            in_edges[head].append(e)
        self.out_edges = tuple(map(tuple, out_edges))
        self.in_edges = tuple(map(tuple, in_edges))
        self.lane_edges = tuple(tuple(path) for path in lanes["edges"])
        self.lane_chokepoints = tuple(tuple(path) for path in lanes["chokepoints"])
        for li, path in enumerate(self.lane_edges):
            nodes = self._path_nodes(path)
            if tuple(node for node in nodes if node in positions) != self.lane_chokepoints[li]:
                raise ValueError(f"lane {li}: chokepoints do not match ordered edge path")

        routes = []
        edge_users = [[] for _ in self.edge_names]
        choke_users, pool_users = defaultdict(list), defaultdict(list)
        from_slots, to_slots = defaultdict(list), defaultdict(list)
        for s, (e, k, li) in enumerate(zip(slots["edge"], slots["k"], slots["lane"], strict=True)):
            e = _checked_index(e, len(self.edge_names), "slot edge")
            k = _checked_index(k, len(self.commodity_names), "slot commodity")
            if li is None:
                path, chokes = (e,), ()
                if self.edge_tail[e] in positions or self.edge_head[e] in positions:
                    raise ValueError(f"slot {s}: a chokepoint route requires a lane")
            else:
                li = _checked_index(li, len(self.lane_names), "slot lane")
                path, chokes = self.lane_edges[li], self.lane_chokepoints[li]
                if e != path[0]:
                    raise ValueError(f"slot {s}: edge must be the first edge of its lane")
            if any(k not in edges["K"][pe] for pe in path):
                raise ValueError(f"slot {s}: commodity is not allowed on every route edge")
            nodes = self._path_nodes(path)
            route = SlotRoute(
                slot_id=s,
                commodity_id=k,
                source_node=nodes[0],
                destination_node=nodes[-1],
                edge_id=e,
                lane_id=li,
                edges=path,
                nodes=nodes,
                chokepoints=chokes,
                chokepoint_positions=tuple(positions[c] for c in chokes),
                pool=goods["pool"][k],
                unit=static["units"][self.commodity_names[k]],
                nominal_transit_weeks=sum(self.edge_transit_weeks[pe] for pe in path),
                nominal_freight_per_unit=math.fsum(edges["c0"][pe] for pe in path),
            )
            routes.append(route)
            from_slots[route.source_node, k].append(s)
            to_slots[route.destination_node, k].append(s)
            for pe in path:
                edge_users[pe].append(s)
            for c in chokes:
                choke_users[c].append(s)
                pool_users[c, route.pool].append(s)
        shape = tuple(config["spaces"]["action"]["flows"]["shape"])
        if shape != (len(routes),):
            raise ValueError("action slots do not match flows shape")
        self.routes = tuple(routes)
        self.edges_to_slots = tuple(map(tuple, edge_users))
        self.chokepoints_to_slots = _freeze_groups(choke_users)
        self.chokepoint_pools_to_slots = _freeze_groups(pool_users)
        self.slots_from = _freeze_groups(from_slots)
        self.slots_to = _freeze_groups(to_slots)

    def _path_nodes(self, path):
        if not path or len(set(path)) != len(path):
            raise ValueError("route must contain a nonempty path without repeated edges")
        for e in path:
            _checked_index(e, len(self.edge_names), "route edge")
        if any(self.edge_head[a] != self.edge_tail[b] for a, b in zip(path, path[1:])):
            raise ValueError("route edges must form a continuous directed path")
        return (self.edge_tail[path[0]],) + tuple(self.edge_head[e] for e in path)

    def shared_resources(self, first_slot, second_slot):
        a = self.routes[_checked_index(first_slot, len(self.routes), "slot")]
        b = self.routes[_checked_index(second_slot, len(self.routes), "slot")]
        edges = tuple(e for e in a.edges if e in b.edges)
        chokes = tuple(c for c in a.chokepoints if c in b.chokepoints)
        pools = tuple((c, a.pool) for c in chokes) if a.pool == b.pool else ()
        return SharedResources(edges, chokes, pools)

    def transit_progress(self, edge_id, lane_id):
        """Resolve a known pipeline edge/lane, never an observed-mask padding row.

        Pass None only for a shipment confirmed off a lane. Hidden lane data
        must be treated as unknown by the caller, not inferred to be None.
        """
        e = _checked_index(edge_id, len(self.edge_names), "pipeline edge")
        if lane_id is None:
            if self.edge_tail[e] in self.chokepoints or self.edge_head[e] in self.chokepoints:
                raise ValueError("a shipment at a chokepoint needs a known lane")
            remaining, chokes, destination = (), (), self.edge_head[e]
        else:
            lane_id = _checked_index(lane_id, len(self.lane_names), "pipeline lane")
            path = self.lane_edges[lane_id]
            if e not in path:
                raise ValueError("pipeline edge is not on its lane")
            remaining = path[path.index(e) + 1 :]
            destination = self.edge_head[path[-1]]
            remaining_nodes = (self.edge_head[e],) + tuple(self.edge_head[pe] for pe in remaining)
            chokes = tuple(node for node in remaining_nodes if node in self.chokepoints)
        return TransitProgress(
            e,
            lane_id,
            self.edge_head[e],
            destination,
            remaining,
            chokes,
            sum(self.edge_transit_weeks[pe] for pe in remaining),
        )


class DataSource(IntEnum):
    CURRENT = 0
    HISTORY = 1
    NOMINAL = 2
    DERIVED = 3


def _readonly(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class FieldEstimate:
    values: np.ndarray
    observed: np.ndarray
    source: np.ndarray
    age_weeks: np.ndarray  # -1 means no remembered observation is being used.
    nominal: np.ndarray


@dataclass(frozen=True, slots=True)
class RouteStatus:
    slot_id: int
    sanction_allowed: bool
    permission_observed: bool
    entry_capacity: float
    snapshot_throughput: float  # Shared downstream resources are not reserved.
    zero_capacity_edges: tuple[int, ...]
    closed_chokepoints: tuple[int, ...]
    uncertain_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NetworkSnapshot:
    week: int
    horizon: int
    fields: object  # Read-only mapping of observation key -> FieldEstimate.
    routes: tuple[RouteStatus, ...]


class NetworkTracker:
    """V2: current observations, bounded history and labelled nominal estimates.

    Hidden zeroes are ignored. Recent last observations persist for at most
    max_history_age weeks; older or never-seen entries use nominal assumptions.
    Estimates are never labelled as current confirmed observations.
    """

    def __init__(self, config, network=None, max_history_age=4):
        self.network = network if network is not None else StaticNetwork(config)
        self.horizon = int(config["T"])
        self.max_history_age = index(max_history_age)
        if self.max_history_age < 0:
            raise ValueError("max_history_age must be nonnegative")
        static, net = config["static"], self.network
        E, K, C = len(net.edge_names), len(net.commodity_names), len(net.chokepoints)
        prohibited = np.zeros((E, K), dtype=float)
        edge_index = {name: e for e, name in enumerate(net.edge_names)}
        commodity_index = {name: k for k, name in enumerate(net.commodity_names)}
        for pair in static["instance"]["prohibitions_at_reset"]:
            prohibited[edge_index[pair["edge"]], commodity_index[pair["k"]]] = 1
        kappas = {pool: [] for pool in ("tb", "ct")}
        for node in net.chokepoints:
            attrs = static["instance"]["nodes"][node]["chokepoint"]
            for pool in kappas:
                kappas[pool].append(attrs["k_c"] * attrs["mu"][pool])
        self.nominal = {
            "graph_now.u": np.array([np.nan if u is None else u for u in net.edge_capacity]),
            "graph_now.c": np.array(static["edges"]["c0"], dtype=float),
            "graph_now.tau": np.array(net.edge_transit_weeks, dtype=float),
            "graph_now.prohibited": prohibited,
            "graph_now.tariff": np.zeros((E, K)),
            "graph_now.open": np.ones(C),
            "graph_now.kappa.tb": np.array(kappas["tb"], dtype=float),
            "graph_now.kappa.ct": np.array(kappas["ct"], dtype=float),
            "graph_now.war_risk": np.zeros(C),
            "action_mask": np.array(
                [not any(prohibited[e, r.commodity_id] for e in r.edges) for r in net.routes], dtype=float
            ),
        }
        self.nominal = MappingProxyType({key: _readonly(value) for key, value in self.nominal.items()})
        self._history = {
            key: (value.copy(), np.full(value.shape, -1, dtype=int)) for key, value in self.nominal.items()
        }
        self._last_week = 0

    def _resolve(self, observation, key, week):
        nominal = self.nominal[key]
        values = np.asarray(observation[key], dtype=float)
        mask = np.asarray(observation[key + ".observed"])
        expected_mask_shape = (1,) if key == "action_mask" else nominal.shape
        if values.shape != nominal.shape or mask.shape != expected_mask_shape:
            raise ValueError(f"{key}: observation or mask shape does not match config")
        if not np.all(np.isin(mask, (0, 1))):
            raise ValueError(f"{key}: observed mask must contain only 0/1")
        seen = np.broadcast_to(mask == 1, nominal.shape)
        shown = values[seen]
        if not np.all(np.isfinite(shown)) or np.any(shown < 0):
            raise ValueError(f"{key}: observed values must be finite and nonnegative")
        if key in ("action_mask", "graph_now.prohibited") and not np.all(np.isin(shown, (0, 1))):
            raise ValueError(f"{key}: observed flags must contain only 0/1")
        if key == "graph_now.open" and np.any(shown > 1):
            raise ValueError("graph_now.open: observed fractions must be in [0, 1]")
        if key in ("graph_now.tau", "graph_now.war_risk") and np.any(shown != np.floor(shown)):
            raise ValueError(f"{key}: observed values must be integers")
        if key == "graph_now.war_risk" and np.any(shown > 2):
            raise ValueError("graph_now.war_risk: unknown class code")
        remembered, last_seen = (a.copy() for a in self._history[key])
        remembered[seen], last_seen[seen] = values[seen], week
        recent = (last_seen >= 1) & (week - last_seen <= self.max_history_age)
        selected = np.where(recent, remembered, nominal)
        source = np.where(seen, DataSource.CURRENT, np.where(recent, DataSource.HISTORY, DataSource.NOMINAL))
        age = np.where(recent, week - last_seen, -1)
        estimate = FieldEstimate(*map(_readonly, (selected, seen, source, age, nominal)))
        return estimate, (remembered, last_seen)

    def update(self, observation):
        week_values = np.asarray(observation["week"])
        if week_values.shape != (1,):
            raise ValueError("week must have shape (1,)")
        week = _checked_index(index(week_values[0]) - 1, self.horizon, "week") + 1
        if week <= self._last_week:
            raise ValueError("update must be called once per week, in increasing order")
        fields, history = {}, {}
        for key in self.nominal:
            fields[key], history[key] = self._resolve(observation, key, week)
        # kappa already includes open. When kappa is hidden, derive it from the
        # resolved openness and the public k_c*mu formula, labelling the estimate.
        openness = fields["graph_now.open"]
        for pool in ("tb", "ct"):
            key = f"graph_now.kappa.{pool}"
            field = fields[key]
            values = np.where(field.observed, field.values, field.nominal * openness.values)
            source = np.where(
                field.observed,
                DataSource.CURRENT,
                np.where(openness.source == DataSource.NOMINAL, DataSource.NOMINAL, DataSource.DERIVED),
            )
            age = np.where(field.observed, 0, openness.age_weeks)
            fields[key] = FieldEstimate(*map(_readonly, (values, field.observed, source, age, field.nominal)))
        statuses = []
        for route in self.network.routes:
            es, k, ps = list(route.edges), route.commodity_id, list(route.chokepoint_positions)
            mask = fields["action_mask"]
            z = fields["graph_now.prohibited"]
            shown_prohibition = bool(np.any(z.observed[es, k] & (z.values[es, k] == 1)))
            if shown_prohibition or (mask.observed[route.slot_id] and mask.values[route.slot_id] == 0):
                allowed, confirmed = False, True
            elif mask.observed[route.slot_id] or np.all(z.observed[es, k]):
                allowed, confirmed = True, True
            else:
                allowed = bool(mask.values[route.slot_id] and not np.any(z.values[es, k]))
                confirmed = False
            u = fields["graph_now.u"].values
            kappa = fields[f"graph_now.kappa.{route.pool}"].values
            closed = tuple(c for c, p in zip(route.chokepoints, ps) if openness.values[p] == 0)
            throughput = min([float(u[e]) for e in es] + [float(kappa[p]) for p in ps])
            if closed:
                throughput = 0.0
            relevant = {
                "graph_now.u": (es,),
                "graph_now.c": (es,),
                "graph_now.tau": (es,),
                "graph_now.prohibited": (es, k),
                "graph_now.tariff": (es, k),
                "graph_now.open": (ps,),
                f"graph_now.kappa.{route.pool}": (ps,),
                "graph_now.war_risk": (ps,),
                "action_mask": (route.slot_id,),
            }
            uncertain = tuple(key for key, indices in relevant.items() if not np.all(fields[key].observed[indices]))
            statuses.append(
                RouteStatus(
                    route.slot_id,
                    allowed,
                    confirmed,
                    float(u[route.edge_id]),
                    throughput,
                    tuple(e for e in es if u[e] == 0),
                    closed,
                    uncertain,
                )
            )
        snapshot = NetworkSnapshot(week, self.horizon, MappingProxyType(fields), tuple(statuses))
        self._history, self._last_week = history, week
        return snapshot
