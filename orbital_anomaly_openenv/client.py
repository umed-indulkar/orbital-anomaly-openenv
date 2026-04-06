# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Orbital Satellite Anomaly Response Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    OrbitalAnomalyOpenenvAction,
    OrbitalAnomalyOpenenvObservation,
)


class OrbitalAnomalyOpenenvEnv(
    EnvClient[OrbitalAnomalyOpenenvAction, OrbitalAnomalyOpenenvObservation, State]
):
    """
    Typed client for the satellite anomaly response environment.
    """

    def _step_payload(self, action: OrbitalAnomalyOpenenvAction) -> Dict:
        return {
            "action_type": action.action_type,
        }

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[OrbitalAnomalyOpenenvObservation]:
        obs_data = payload.get("observation", {})

        observation = OrbitalAnomalyOpenenvObservation(
            battery_level=obs_data.get("battery_level", 0.0),
            solar_efficiency=obs_data.get("solar_efficiency", 0.0),
            thermal_temp=obs_data.get("thermal_temp", 0.0),
            comms_signal=obs_data.get("comms_signal", 0.0),
            payload_on=obs_data.get("payload_on", False),
            safe_mode=obs_data.get("safe_mode", False),
            task_id=obs_data.get("task_id", "easy"),
            mission_status=obs_data.get("mission_status", "stable"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )