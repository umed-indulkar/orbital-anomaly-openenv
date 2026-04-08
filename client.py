# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Orbital Satellite Anomaly Response — typed OpenEnv client (V2)."""

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
    Typed synchronous/async client for the satellite anomaly environment V2.

    Fully backward compatible with V1 graders — all new fields have safe
    defaults and are parsed from the server response when present.
    """

    def _reset_payload(self, task_id: Optional[str] = None) -> Dict:
        payload: Dict = {}
        if task_id is not None:
            payload["task_id"] = task_id
        return payload

    def _step_payload(self, action: OrbitalAnomalyOpenenvAction) -> Dict:
        return {"action_type": action.action_type}

    def _parse_result(self, payload: Dict) -> StepResult[OrbitalAnomalyOpenenvObservation]:
        obs_data = payload.get("observation", {})
        observation = self._build_observation(obs_data, payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward") or obs_data.get("reward"),
            done=payload.get("done", False),
        )

    def _build_observation(self, obs: Dict, top: Dict) -> OrbitalAnomalyOpenenvObservation:
        """Safely construct observation, tolerating absent V2 fields."""
        return OrbitalAnomalyOpenenvObservation(
            # ── V1 fields ─────────────────────────────────────────────
            battery_level=obs.get("battery_level", 100.0),
            solar_efficiency=obs.get("solar_efficiency", 1.0),
            thermal_temp=obs.get("thermal_temp", 40.0),
            comms_signal=obs.get("comms_signal", 1.0),
            payload_on=obs.get("payload_on", True),
            safe_mode=obs.get("safe_mode", False),
            task_id=obs.get("task_id", "easy"),
            mission_status=obs.get("mission_status", "stable"),
            reward=top.get("reward") or obs.get("reward"),
            done=top.get("done", False),
            metadata=obs.get("metadata", {}),

            # ── V2 EPS ────────────────────────────────────────────────
            battery_soc=obs.get("battery_soc", obs.get("battery_level", 85.0)),
            bus_voltage=obs.get("bus_voltage", 28.0),
            panel_health=obs.get("panel_health", 1.0),
            solar_array_current=obs.get("solar_array_current", -1.0),
            charge_controller_health=obs.get("charge_controller_health", 1.0),
            power_bus_redundancy=obs.get("power_bus_redundancy", True),

            # ── V2 ADCS ───────────────────────────────────────────────
            attitude_error_deg=obs.get("attitude_error_deg", 5.0),
            sun_vector_alignment=obs.get("sun_vector_alignment", 0.996),
            reaction_wheel_momentum=obs.get("reaction_wheel_momentum", 0.1),
            gyro_bias=obs.get("gyro_bias", 0.0),
            star_tracker_available=obs.get("star_tracker_available", True),
            wheel_saturation_level=obs.get("wheel_saturation_level", 0.1),

            # ── V2 Thermal ────────────────────────────────────────────
            battery_temp=obs.get("battery_temp", 15.0),
            payload_temp=obs.get("payload_temp", obs.get("thermal_temp", 25.0)),
            avionics_temp=obs.get("avionics_temp", 30.0),
            radiator_efficiency=obs.get("radiator_efficiency", 1.0),
            heater_state=obs.get("heater_state", False),
            thermal_loop_health=obs.get("thermal_loop_health", 1.0),

            # ── V2 Communications ─────────────────────────────────────
            antenna_pointing_error=obs.get("antenna_pointing_error", 3.0),
            transmitter_power=obs.get("transmitter_power", 5.0),
            bit_error_rate=obs.get("bit_error_rate", 0.001),
            uplink_margin=obs.get("uplink_margin", 12.0),
            packet_loss_ratio=obs.get("packet_loss_ratio", 0.02),
            command_latency_ms=obs.get("command_latency_ms", 120.0),

            # ── V2 Orbital context ────────────────────────────────────
            sunlit=obs.get("sunlit", True),
            eclipse_timer=obs.get("eclipse_timer", 0),
            ground_station_visible=obs.get("ground_station_visible", True),
            radiation_zone=obs.get("radiation_zone", False),
            observation_window_active=obs.get("observation_window_active", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )