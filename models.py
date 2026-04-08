# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Typed action and observation schemas for the
Orbital Satellite Anomaly Response benchmark — Version 2.

V1 fields (battery_level, solar_efficiency, thermal_temp, comms_signal,
payload_on, safe_mode, task_id, mission_status, reward, done, metadata)
are preserved exactly for grader backward compatibility.

V2 fields add full EPS, ADCS, multi-zone thermal, RF comms chain,
and orbital context telemetry.
"""

from typing import Any, Dict, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class OrbitalAnomalyOpenenvAction(Action):
    """
    Mission-control action space for spacecraft recovery.

    The same 6-action space is preserved from V1 for grader compatibility.
    Internally each action now affects multiple V2 subsystems.
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
            "Corrective mission-control action applied to spacecraft subsystems.\n"
            "rotate_to_sun     : ADCS attitude correction toward sun vector — "
            "improves solar charging, reduces antenna pointing error.\n"
            "disable_payload   : Shut down science payload — reduces thermal "
            "and power load immediately.\n"
            "reboot_comms      : RF chain reset — reduces BER, packet loss, "
            "latency; may clear transponder overheating fault if caught early.\n"
            "enter_safe_mode   : Conservative hold — disables payload, reduces "
            "wheel stress, cools all zones; sacrifices science throughput.\n"
            "switch_power_bus  : Activate redundant power bus — injects battery "
            "reserve, clears bus short fault, raises bus voltage.\n"
            "noop              : Take no action this step."
        ),
    )


class OrbitalAnomalyOpenenvObservation(Observation):
    """
    Spacecraft telemetry observation returned after each control step.

    Combines V1 backward-compatible fields with V2 subsystem telemetry.
    Some fields may carry sentinel values (-999, -1, None) when the
    corresponding sensor is in dropout due to a fault or orbital blackout —
    the agent must handle missing data gracefully.
    """

    # ── V1 backward-compatible fields (grader stability) ─────────────
    battery_level: float = Field(
        default=100.0,
        description="Battery state-of-charge [0, 100]. Same as battery_soc.",
    )
    solar_efficiency: float = Field(
        default=1.0,
        description="Effective solar generation factor [0, 1] = sun_vector * panel_health.",
    )
    thermal_temp: float = Field(
        default=40.0,
        description="Payload zone temperature [°C]. Same as payload_temp.",
    )
    comms_signal: float = Field(
        default=1.0,
        description="Composite comms quality [0, 1] derived from BER + packet loss.",
    )
    payload_on: bool = Field(
        default=True,
        description="True when science payload is active.",
    )
    safe_mode: bool = Field(
        default=False,
        description="True when spacecraft is in safe/survival mode.",
    )
    task_id: Literal["easy", "medium", "hard"] = Field(
        default="easy",
        description="Active benchmark task difficulty.",
    )
    mission_status: Literal["stable", "warning", "critical"] = Field(
        default="stable",
        description="Overall mission health classification.",
    )
    reward: Optional[float] = Field(
        default=None,
        description="Per-step reward in strict open interval (0, 1).",
    )
    done: bool = Field(
        default=False,
        description="True when episode terminates.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary info: step, episode_id, version, obs_dropout list.",
    )

    # ── V2 EPS telemetry ──────────────────────────────────────────────
    battery_soc: float = Field(
        default=85.0,
        description="Battery state-of-charge [0, 100 %].",
    )
    bus_voltage: float = Field(
        default=28.0,
        description="Spacecraft bus voltage [18–28 V]. Nominal = 28 V.",
    )
    panel_health: float = Field(
        default=1.0,
        description="Solar panel health multiplier [0, 1]. Degrades in radiation zones.",
    )
    solar_array_current: float = Field(
        default=4.0,
        description=(
            "Solar array output current [A]. "
            "-1.0 indicates telemetry not available this step."
        ),
    )
    charge_controller_health: float = Field(
        default=1.0,
        description="MPPT charge controller health [0, 1]. Reduced by stuck MPPT fault.",
    )
    power_bus_redundancy: bool = Field(
        default=True,
        description="True when redundant power bus is active.",
    )

    # ── V2 ADCS telemetry ─────────────────────────────────────────────
    attitude_error_deg: float = Field(
        default=5.0,
        description="Spacecraft attitude error from optimal sun-pointing [deg, 0–90].",
    )
    sun_vector_alignment: float = Field(
        default=0.996,
        description="cos(attitude_error) — solar incidence factor [0, 1].",
    )
    reaction_wheel_momentum: float = Field(
        default=0.1,
        description="Normalised reaction wheel angular momentum [0, 1]. 1.0 = saturated.",
    )
    gyro_bias: float = Field(
        default=0.0,
        description=(
            "Accumulated gyro drift bias [deg/s]. "
            "-999.0 indicates telemetry not available this step."
        ),
    )
    star_tracker_available: Optional[bool] = Field(
        default=True,
        description="Star tracker operational status. None = sensor in dropout.",
    )
    wheel_saturation_level: float = Field(
        default=0.1,
        description="Reaction wheel saturation level [0, 1]. >0.8 severely limits ADCS.",
    )

    # ── V2 Multi-zone thermal telemetry ───────────────────────────────
    battery_temp: float = Field(
        default=15.0,
        description="Battery compartment temperature [°C]. Safe range: -5 to 35.",
    )
    payload_temp: float = Field(
        default=25.0,
        description="Payload module temperature [°C]. Safe range: <75.",
    )
    avionics_temp: float = Field(
        default=30.0,
        description=(
            "Avionics bay temperature [°C]. Safe range: <70. "
            "-999.0 indicates telemetry not available this step."
        ),
    )
    radiator_efficiency: float = Field(
        default=1.0,
        description="Thermal radiator effectiveness [0, 1]. Reduced by stuck radiator fault.",
    )
    heater_state: bool = Field(
        default=False,
        description="Battery heater currently energised.",
    )
    thermal_loop_health: float = Field(
        default=1.0,
        description="Coolant loop / heat pipe health [0, 1].",
    )

    # ── V2 RF Communications telemetry ───────────────────────────────
    antenna_pointing_error: float = Field(
        default=3.0,
        description="Antenna boresight pointing error [deg]. Driven by attitude error.",
    )
    transmitter_power: float = Field(
        default=5.0,
        description="RF transmitter output power [W]. Degrades with amplifier fault.",
    )
    bit_error_rate: float = Field(
        default=0.001,
        description="Link bit error rate [0, 1]. <0.01 = good link quality.",
    )
    uplink_margin: float = Field(
        default=12.0,
        description=(
            "Uplink link margin [dB]. -99.0 indicates telemetry not available "
            "(ground station blackout)."
        ),
    )
    packet_loss_ratio: float = Field(
        default=0.02,
        description="Fraction of packets lost [0, 1]. <0.05 = good.",
    )
    command_latency_ms: float = Field(
        default=120.0,
        description=(
            "Round-trip command latency [ms]. "
            "-1.0 indicates telemetry not available this step."
        ),
    )

    # ── V2 Orbital context ────────────────────────────────────────────
    sunlit: bool = Field(
        default=True,
        description="True when spacecraft is in sunlight (solar charging possible).",
    )
    eclipse_timer: int = Field(
        default=0,
        description="Steps elapsed since eclipse entry. 0 if currently sunlit.",
    )
    ground_station_visible: bool = Field(
        default=True,
        description="True when a ground station is in view for uplink/downlink.",
    )
    radiation_zone: bool = Field(
        default=False,
        description="True when spacecraft is transiting a radiation belt.",
    )
    observation_window_active: bool = Field(
        default=False,
        description=(
            "True during active science observation window. "
            "Payload uptime during this window contributes science bonus reward."
        ),
    )