# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Orbital Satellite Anomaly Response — typed OpenEnv client."""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import (
    OrbitalAnomalyOpenenvAction,
    OrbitalAnomalyOpenenvObservation,
)


class OrbitalAnomalyOpenenvEnv(
    EnvClient[OrbitalAnomalyOpenenvAction, OrbitalAnomalyOpenenvObservation, State]
):
    """
    Typed synchronous/async client for the satellite anomaly environment.

    Supports an optional ``task_id`` parameter on ``reset()`` so the
    Phase-2 grader can target specific benchmark tasks directly.
    """

    # ── Reset ─────────────────────────────────────────────────────────

    def _reset_payload(self, task_id: Optional[str] = None) -> Dict:
        """Build the JSON body for POST /reset."""
        payload: Dict = {}
        if task_id is not None:
            payload["task_id"] = task_id
        return payload

    # ── Step ──────────────────────────────────────────────────────────

    def _step_payload(self, action: OrbitalAnomalyOpenenvAction) -> Dict:
        return {"action_type": action.action_type}

    # ── Response parsing ──────────────────────────────────────────────

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[OrbitalAnomalyOpenenvObservation]:
        obs_data = payload.get("observation", {})

        observation = OrbitalAnomalyOpenenvObservation(
            battery_level=obs_data.get("battery_level", 100.0),
            solar_efficiency=obs_data.get("solar_efficiency", 1.0),
            thermal_temp=obs_data.get("thermal_temp", 40.0),
            comms_signal=obs_data.get("comms_signal", 1.0),
            payload_on=obs_data.get("payload_on", True),
            safe_mode=obs_data.get("safe_mode", False),
            task_id=obs_data.get("task_id", "easy"),
            mission_status=obs_data.get("mission_status", "stable"),
            reward=payload.get("reward") or obs_data.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward") or obs_data.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )