# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Typed action and observation schemas for the
Orbital Satellite Anomaly Response benchmark.
"""

from typing import Any, Dict, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class OrbitalAnomalyOpenenvAction(Action):
    """
    Mission-control action space for spacecraft recovery.

    The agent selects one corrective command per step.
    """

    action_type: Literal[
        "rotate_to_sun",
        "disable_payload",
        "reboot_comms",
        "enter_safe_mode",
        "switch_power_bus",
        "noop",
    ] = Field(
        ...,
        description=(
            "Corrective mission-control action applied to spacecraft subsystems. "
            "rotate_to_sun: improve solar charging efficiency. "
            "disable_payload: reduce thermal/power load. "
            "reboot_comms: restore communication signal. "
            "enter_safe_mode: stabilise all subsystems conservatively. "
            "switch_power_bus: inject emergency battery reserves. "
            "noop: take no action this step."
        ),
    )


class OrbitalAnomalyOpenenvObservation(Observation):
    """
    Spacecraft telemetry observation returned after each control step.

    All continuous values are normalised to their documented ranges so
    agents can directly compare subsystem health signals.
    """

    # ── Continuous telemetry ──────────────────────────────────────────
    battery_level: float = Field(
        default=100.0,
        description="Current spacecraft battery percentage [0, 100].",
    )
    solar_efficiency: float = Field(
        default=1.0,
        description="Solar charging efficiency [0, 1].",
    )
    thermal_temp: float = Field(
        default=40.0,
        description="Payload thermal temperature in Celsius [0, 120].",
    )
    comms_signal: float = Field(
        default=1.0,
        description="Communication signal strength [0, 1].",
    )

    # ── Discrete flags ────────────────────────────────────────────────
    payload_on: bool = Field(
        default=True,
        description="Whether the science payload is currently active.",
    )
    safe_mode: bool = Field(
        default=False,
        description="Whether the spacecraft is operating in safe mode.",
    )

    # ── Episode metadata ──────────────────────────────────────────────
    task_id: Literal["easy", "medium", "hard"] = Field(
        default="easy",
        description="Active benchmark task difficulty identifier.",
    )
    mission_status: Literal["stable", "warning", "critical"] = Field(
        default="stable",
        description="Overall mission health classification derived from telemetry.",
    )

    # ── Step signals ──────────────────────────────────────────────────
    reward: Optional[float] = Field(
        default=None,
        description=(
            "Per-step reward in the strict open interval (0, 1). "
            "Combines battery health, inverse thermal load, and comms quality."
        ),
    )
    done: bool = Field(
        default=False,
        description=(
            "True when the episode terminates — either the satellite is fully "
            "stabilised or the step budget (12 steps) is exhausted."
        ),
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary information: step count, episode_id, etc.",
    )