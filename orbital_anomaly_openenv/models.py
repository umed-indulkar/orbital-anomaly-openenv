# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import Literal
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class OrbitalAnomalyOpenenvAction(Action):
    """
    Action model for spacecraft anomaly response.
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
        description="Mission control corrective action to apply",
    )


class OrbitalAnomalyOpenenvObservation(Observation):
    """
    Telemetry observation returned from spacecraft simulator.
    """

    battery_level: float = Field(
        default=100.0,
        description="Current spacecraft battery percentage (0-100)",
    )

    solar_efficiency: float = Field(
        default=1.0,
        description="Solar charging efficiency (0-1)",
    )

    thermal_temp: float = Field(
        default=40.0,
        description="Payload thermal temperature in Celsius",
    )

    comms_signal: float = Field(
        default=1.0,
        description="Communication signal strength (0-1)",
    )

    payload_on: bool = Field(
        default=True,
        description="Whether science payload is active",
    )

    safe_mode: bool = Field(
        default=False,
        description="Whether spacecraft is in safe mode",
    )

    task_id: Literal["easy", "medium", "hard"] = Field(
        default="easy",
        description="Current anomaly scenario difficulty",
    )

    mission_status: Literal["stable", "warning", "critical"] = Field(
        default="stable",
        description="Overall mission condition",
    )