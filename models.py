# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Typed action and observation schemas for the
Orbital Satellite Anomaly Response benchmark.
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class OrbitalAnomalyOpenenvAction(Action):
    """
    Mission-control action space for spacecraft recovery.
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
        description="Corrective mission-control action applied to spacecraft subsystems.",
    )


class OrbitalAnomalyOpenenvObservation(Observation):
    """
    Spacecraft telemetry observation returned after each control step.
    """

    battery_level: float = Field(
        default=100.0,
        description="Current spacecraft battery percentage in range [0, 100].",
    )

    solar_efficiency: float = Field(
        default=1.0,
        description="Solar charging efficiency in range [0, 1].",
    )

    thermal_temp: float = Field(
        default=40.0,
        description="Payload thermal temperature in Celsius.",
    )

    comms_signal: float = Field(
        default=1.0,
        description="Communication signal strength in range [0, 1].",
    )

    payload_on: bool = Field(
        default=True,
        description="Whether the science payload is currently active.",
    )

    safe_mode: bool = Field(
        default=False,
        description="Whether the spacecraft is operating in safe mode.",
    )

    task_id: Literal["easy", "medium", "hard"] = Field(
        default="easy",
        description="Deterministic benchmark task difficulty.",
    )

    mission_status: Literal["stable", "warning", "critical"] = Field(
        default="stable",
        description="Overall mission health classification.",
    )