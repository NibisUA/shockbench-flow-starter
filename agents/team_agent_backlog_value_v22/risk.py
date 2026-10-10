"""Markian's Risk Analysis Module (Task M5).

Processes early warning signals, messages, pending sanctions, and closure announcements:
- De-duplicates announcement threads across weeks using msg_id (handles withdrawals, proposals, final notices).
- Tracks warning score history and computes rising/falling trend signals.
- Computes calibrated risk scores per chokepoint, edge, region, and network-wide.
- Directly feeds into routing decisions, queue forecasts, and safety buffer sizing.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

try:
    from risk_contracts import RiskState
except ImportError:
    from .risk_contracts import RiskState


class RiskAnalyzer:
    """Evaluates multi-source disruption risks and maintains signal history."""

    def __init__(self, config: Dict[str, Any]):
        self.static = config["static"]
        self.layout = config["layout"]
        self.spaces = config["spaces"]

        # Chokepoints mapping
        self.chokepoints = list(self.layout.get("chokepoints", []))
        self.n_chokepoints = len(self.chokepoints)
        self.chk_pos_to_node = {i: node for i, node in enumerate(self.chokepoints)}
        self.chk_node_to_pos = {node: i for i, node in enumerate(self.chokepoints)}

        # Warning units mapping
        self.warning_units = list(self.layout.get("warning_units", []))
        self.n_warnings = len(self.warning_units)

        # Edges count
        self.n_edges = len(self.static["edges"]["tail"])

        # History tracking (Roadmap M5)
        self.warning_history: List[np.ndarray] = []
        self.seen_thread_ids: Set[int] = set()
        self.withdrawn_thread_ids: Set[int] = set()
        self.active_threads_by_target: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

    def analyze_risks(self, observation: Dict[str, np.ndarray]) -> RiskState:
        """Process this week's observation and produce updated RiskState (Roadmap M5)."""
        current_week = int(observation["week"][0])

        # -------------------------------------------------------------
        # 1. Early-Warning Scores & Trend Detection
        # -------------------------------------------------------------
        w_obs = observation["warning.score.observed"]
        raw_warnings = observation["warning.score"]
        current_warnings = np.where(w_obs == 1, raw_warnings, 0.0)

        self.warning_history.append(current_warnings.copy())
        if len(self.warning_history) > 6:
            self.warning_history.pop(0)

        # Compute trend over past 3 weeks (rising trend = acute hazard)
        warning_trend = np.zeros_like(current_warnings)
        if len(self.warning_history) >= 3:
            old_w = self.warning_history[-3]
            warning_trend = np.clip(current_warnings - old_w, 0.0, 1.0)

        # -------------------------------------------------------------
        # 2. De-duplicate Announcements (messages.* using msg_id)
        # -------------------------------------------------------------
        self._process_messages(observation, current_week)

        # -------------------------------------------------------------
        # 3. Pending Prohibitions / Sanctions
        # -------------------------------------------------------------
        pending_list = self._process_pending_prohibitions(observation, current_week)

        # -------------------------------------------------------------
        # 4. Strait Closures & Reopening End-Weeks
        # -------------------------------------------------------------
        closure_forecasts = self._process_closure_announcements(observation)

        # -------------------------------------------------------------
        # 5. Synthesize Per-Chokepoint Risk Scores
        # -------------------------------------------------------------
        chokepoint_risk: Dict[int, float] = {}
        open_fracs = observation["graph_now.open"]
        open_obs = observation["graph_now.open.observed"]

        for pos, chk_node in self.chk_pos_to_node.items():
            # Current physical openness
            o_c = float(open_fracs[pos]) if open_obs[pos] == 1 else 1.0
            direct_closure_risk = np.clip(1.0 - o_c, 0.0, 1.0)

            # Warning signal for chokepoint (located at the end of warning_units layout)
            chk_warn_idx = self.n_warnings - self.n_chokepoints + pos
            warn_val = float(current_warnings[chk_warn_idx]) if chk_warn_idx < self.n_warnings else 0.0
            trend_val = float(warning_trend[chk_warn_idx]) if chk_warn_idx < self.n_warnings else 0.0

            # Threat messages targeting this chokepoint (target_kind 0)
            target_threats = self.active_threads_by_target.get((0, chk_node), [])
            msg_risk = min(len(target_threats) * 0.25, 0.75)

            # Composite chokepoint risk score
            c_risk = (
                0.35 * direct_closure_risk
                + 0.35 * warn_val
                + 0.15 * trend_val
                + 0.15 * msg_risk
            )
            chokepoint_risk[chk_node] = float(np.clip(c_risk, 0.0, 1.0))

        # -------------------------------------------------------------
        # 6. Synthesize Per-Edge Risk Scores (Sanctions & Tariffs)
        # -------------------------------------------------------------
        edge_risk: Dict[int, float] = {}
        tariff_mat = observation["graph_now.tariff"]
        tariff_obs = observation["graph_now.tariff.observed"]

        # Check pending sanctions per edge
        edges_with_pending: Dict[int, int] = {}
        for p in pending_list:
            e = p["edge"]
            w_eff = p["effective_week"]
            weeks_left = max(0, w_eff - current_week)
            edges_with_pending[e] = min(edges_with_pending.get(e, 99), weeks_left)

        for e in range(self.n_edges):
            e_risk = 0.0
            # Pending sanction nearing?
            if e in edges_with_pending:
                weeks_left = edges_with_pending[e]
                if weeks_left <= 2:
                    e_risk += 0.85
                elif weeks_left <= 5:
                    e_risk += 0.50
                else:
                    e_risk += 0.25

            # High tariff rate?
            if np.any(tariff_obs[e] == 1):
                max_t = float(np.max(tariff_mat[e]))
                if max_t > 0.2:
                    e_risk += min(max_t * 0.5, 0.4)

            # Messages targeting this edge (target_kind 1)
            edge_threats = self.active_threads_by_target.get((1, e), [])
            e_risk += min(len(edge_threats) * 0.2, 0.4)

            edge_risk[e] = float(np.clip(e_risk, 0.0, 1.0))

        # -------------------------------------------------------------
        # 7. Global Composite Risk
        # -------------------------------------------------------------
        max_chk = max(chokepoint_risk.values()) if chokepoint_risk else 0.0
        avg_warn = float(np.mean(current_warnings)) if len(current_warnings) > 0 else 0.0
        n_pending = len(pending_list)
        pending_factor = min(n_pending / 10.0, 0.5)

        global_risk = float(
            np.clip(
                0.50 * max_chk + 0.25 * avg_warn + 0.25 * pending_factor,
                0.0,
                1.0,
            )
        )

        return RiskState(
            week=current_week,
            chokepoint_risk=chokepoint_risk,
            edge_risk=edge_risk,
            region_risk={},
            global_risk=global_risk,
            active_threat_ids=self.seen_thread_ids - self.withdrawn_thread_ids,
            pending_sanctions=pending_list,
            closure_forecasts=closure_forecasts,
        )

    def _process_messages(self, observation: Dict[str, np.ndarray], current_week: int) -> None:
        """De-duplicate messages by msg_id, tracking cancellations and active threats (Roadmap M5)."""
        m_obs = observation["messages.msg_id.observed"]
        live_indices = np.where(m_obs == 1)[0]

        id_arr = observation["messages.msg_id"]
        kind_arr = observation["messages.kind"]
        target_kind_arr = observation["messages.target_kind"]
        target_arr = observation["messages.target"]
        comm_arr = observation["messages.k"]

        self.active_threads_by_target.clear()

        for idx in live_indices:
            msg_id = int(id_arr[idx])
            kind = int(kind_arr[idx])
            t_kind = int(target_kind_arr[idx])
            t_idx = int(target_arr[idx])
            comm = int(comm_arr[idx])

            # Kind 4 = withdrawal: threat cancelled!
            if kind == 4:
                self.withdrawn_thread_ids.add(msg_id)
                continue

            self.seen_thread_ids.add(msg_id)
            if msg_id in self.withdrawn_thread_ids:
                continue  # Skip withdrawn threats

            thread_dict = {
                "msg_id": msg_id,
                "kind": kind,
                "target_kind": t_kind,
                "target": t_idx,
                "commodity": comm,
                "observed_week": current_week,
            }
            self.active_threads_by_target.setdefault((t_kind, t_idx), []).append(thread_dict)

    def _process_pending_prohibitions(
        self,
        observation: Dict[str, np.ndarray],
        current_week: int,
    ) -> List[Dict[str, Any]]:
        """Extract confirmed future sanctions and prohibition dates."""
        p_obs = observation["pending_prohibitions.edge.observed"]
        live_indices = np.where(p_obs == 1)[0]

        edges = observation["pending_prohibitions.edge"]
        comms = observation["pending_prohibitions.k"]
        eff_weeks = observation["pending_prohibitions.effective_week"]

        pending: List[Dict[str, Any]] = []
        for idx in live_indices:
            e = int(edges[idx])
            k = int(comms[idx])
            w_eff = int(eff_weeks[idx])
            if w_eff >= current_week:
                pending.append({"edge": e, "commodity": k, "effective_week": w_eff})
        return pending

    def _process_closure_announcements(
        self,
        observation: Dict[str, np.ndarray],
    ) -> Dict[int, Optional[int]]:
        """Extract announced reopening weeks for active strait closures."""
        c_obs = observation["closure_end.chokepoint.observed"]
        live_indices = np.where(c_obs == 1)[0]

        chks = observation["closure_end.chokepoint"]
        ends = observation["closure_end.end_week"]

        forecasts: Dict[int, Optional[int]] = {}
        for idx in live_indices:
            chk = int(chks[idx])
            end_w = int(ends[idx]) if ends[idx] >= 0 else None
            forecasts[chk] = end_w
        return forecasts
