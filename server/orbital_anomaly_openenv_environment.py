# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Orbital Satellite Anomaly Response Environment — Version 2.

A high-fidelity spacecraft digital twin that models:
  - Electrical Power System (EPS) with power-balance physics
  - ADCS with cosine-based solar alignment and wheel saturation
  - Multi-zone thermal propagation (battery / payload / avionics)
  - RF communications chain (antenna pointing + transponder health)
  - Orbital context (eclipse, ground-station windows, observation windows)
  - Latent root-cause fault graph with delayed cascading failures
  - Partial observability (sensor dropout under faults / orbital blackout)
  - Multi-objective dense reward in strict open interval (0, 1)

Public API is identical to V1:
  reset(task_id?)  →  OrbitalAnomalyOpenenvObservation
  step(action)     →  OrbitalAnomalyOpenenvObservation
  state            →  State
"""

import math
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    OrbitalAnomalyOpenenvAction,
    OrbitalAnomalyOpenenvObservation,
)

TASK_IDS = ["easy", "medium", "hard"]

# ── Physical constants ────────────────────────────────────────────────────────
_DT          = 1.0
_SOC_FULL    = 100.0
_SOC_MIN     = 0.0
_BUS_NOMINAL = 28.0
_TEMP_SPACE  = -20.0

# ── Fault identifiers (never exposed in observations) ─────────────────────────
_F_MPPT_STUCK        = "mppt_stuck"
_F_PANEL_JAM         = "panel_deployment_jam"
_F_BUS_SHORT         = "bus_short_transient"
_F_BAT_AGING         = "battery_aging"
_F_RW_SATURATION     = "reaction_wheel_saturation"
_F_GYRO_DRIFT        = "gyro_drift"
_F_STAR_TRACKER_DROP = "star_tracker_dropout"
_F_RADIATOR_STUCK    = "radiator_valve_stuck"
_F_HEAT_PIPE_FAIL    = "heat_pipe_failure"
_F_HEATER_LATCH      = "heater_relay_latch"
_F_TRANSPONDER_HOT   = "transponder_overheating"
_F_AMPLIFIER_DEGRADE = "amplifier_degradation"
_F_ANTENNA_STALL     = "antenna_gimbal_stall"


class OrbitalAnomalyOpenenvEnvironment(Environment):
    """
    Version 2 satellite anomaly response simulator.

    Agents receive rich, partially observable spacecraft telemetry and
    must sequence corrective actions across deeply coupled subsystems
    under delayed fault cascades and orbital context dynamics.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Class-level counter for cycling when no explicit task_id is given.
    # Explicit task_id requests NEVER consume or depend on this counter.
    _global_reset_count: int = 0

    # ──────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._init_nominal_state()

    def _init_nominal_state(self):
        """Initialise all subsystem variables to nominal healthy values.
        NOTE: task_id is intentionally NOT set here — it is always set
        explicitly in reset() after this call returns."""

        # ── EPS ───────────────────────────────────────────────────────
        self.battery_soc: float              = 85.0
        self.bus_voltage: float              = _BUS_NOMINAL
        self.panel_health: float             = 1.0
        self.solar_array_current: float      = 4.0
        self.payload_power_draw: float       = 8.0
        self.avionics_draw: float            = 3.5
        self.heater_draw: float              = 0.0
        self.charge_controller_health: float = 1.0
        self.power_bus_redundancy: bool      = True

        # ── ADCS ──────────────────────────────────────────────────────
        self.attitude_error_deg: float       = 5.0
        self.sun_vector_alignment: float     = 0.996
        self.reaction_wheel_momentum: float  = 0.1
        self.gyro_bias: float                = 0.0
        self.star_tracker_available: bool    = True
        self.wheel_saturation_level: float   = 0.1

        # ── Thermal (multi-zone) ──────────────────────────────────────
        self.battery_temp: float             = 15.0
        self.payload_temp: float             = 25.0
        self.avionics_temp: float            = 30.0
        self.radiator_efficiency: float      = 1.0
        self.heater_state: bool              = False
        self.thermal_loop_health: float      = 1.0

        # ── Communications ────────────────────────────────────────────
        self.antenna_pointing_error: float   = 3.0
        self.transmitter_power: float        = 5.0
        self.bit_error_rate: float           = 0.001
        self.uplink_margin: float            = 12.0
        self.packet_loss_ratio: float        = 0.02
        self.command_latency_ms: float       = 120.0

        # ── Orbital context ───────────────────────────────────────────
        self.sunlit: bool                    = True
        self.eclipse_timer: int              = 0
        self.ground_station_visible: bool    = True
        self.radiation_zone: bool            = False
        self.observation_window_active: bool = False
        self._orbit_step: int                = 0

        # ── Mission / payload ─────────────────────────────────────────
        self.payload_on: bool                = True
        self.safe_mode: bool                 = False
        # task_id is NOT reset here — always set after _init_nominal_state() in reset()
        self.task_id: str                    = "easy"

        # ── Active latent fault set ───────────────────────────────────
        self._faults: set                    = set()
        self._fault_timers: dict             = {}

    # ──────────────────────────────────────────────────────────────────
    # OpenEnv Interface
    # ──────────────────────────────────────────────────────────────────

    def reset(self, task_id: str | None = None, **kwargs) -> OrbitalAnomalyOpenenvObservation:
        """
        Reset into a deterministic benchmark task.

        task_id resolution:
          - Explicit task_id argument → use directly, do NOT touch counter
          - No task_id → cycle via counter (easy → medium → hard → …)

        The resolved task_id is saved to a LOCAL variable before
        _init_nominal_state() runs, so the nominal-state reset can never
        overwrite the chosen task.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # ── Resolve task BEFORE _init_nominal_state() ────────────────
        if task_id in TASK_IDS:
            # Explicit request — never advance the cycle counter
            resolved_task = task_id
        else:
            # Cycling fallback — pick from counter, then advance it
            resolved_task = TASK_IDS[self.__class__._global_reset_count % len(TASK_IDS)]
            self.__class__._global_reset_count += 1

        # ── Reset all subsystem state ─────────────────────────────────
        # _init_nominal_state() will set self.task_id = "easy" as a placeholder,
        # but we immediately overwrite it below with resolved_task.
        self._init_nominal_state()

        # ── Apply the resolved task (always overwrites the placeholder) ─
        self.task_id = resolved_task
        self._load_task(resolved_task)

        return self._get_observation(reward=self._safe_reward(0.45), done=False)

    def step(self, action: OrbitalAnomalyOpenenvAction) -> OrbitalAnomalyOpenenvObservation:
        """Execute one mission-control action and advance the simulation."""
        self._state.step_count += 1

        self._advance_orbital_context()
        self._apply_action(action.action_type)
        self._tick_fault_cascades()
        self._eps_update()
        self._adcs_update()
        self._thermal_update()
        self._comms_update()

        reward = self._compute_reward()
        done   = self._check_done()
        return self._get_observation(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    # ──────────────────────────────────────────────────────────────────
    # Task Initialisation
    # ──────────────────────────────────────────────────────────────────

    def _load_task(self, task_id: str) -> None:
        """Load deterministic initial anomaly conditions and latent faults.
        self.task_id must already be set by reset() before calling this."""

        if task_id == "easy":
            # Primary anomaly: ADCS misalignment + low battery + stuck MPPT
            self.battery_soc                = 38.0
            self.bus_voltage                = 25.8
            self.panel_health               = 0.85
            self.charge_controller_health   = 0.90
            self.payload_power_draw         = 8.0
            self.attitude_error_deg         = 42.0
            self.sun_vector_alignment       = math.cos(math.radians(42.0))
            self.reaction_wheel_momentum    = 0.25
            self.wheel_saturation_level     = 0.25
            self.gyro_bias                  = 0.8
            self.star_tracker_available     = True
            self.battery_temp               = 8.0
            self.payload_temp               = 38.0
            self.avionics_temp              = 32.0
            self.radiator_efficiency        = 0.95
            self.thermal_loop_health        = 0.95
            self.antenna_pointing_error     = 18.0
            self.transmitter_power          = 4.5
            self.bit_error_rate             = 0.008
            self.uplink_margin              = 9.0
            self.packet_loss_ratio          = 0.06
            self.command_latency_ms         = 180.0
            self.sunlit                     = True
            self.eclipse_timer              = 0
            self.ground_station_visible     = True
            self.radiation_zone             = False
            self.observation_window_active  = False
            self._orbit_step                = 0
            self.payload_on                 = True
            self.safe_mode                  = False
            self._faults                    = {_F_MPPT_STUCK}
            self._fault_timers              = {_F_MPPT_STUCK: 0}

        elif task_id == "medium":
            # Primary anomaly: thermal overload + comms degradation + active science window
            self.battery_soc                = 55.0
            self.bus_voltage                = 27.2
            self.panel_health               = 0.75
            self.charge_controller_health   = 0.80
            self.payload_power_draw         = 8.0
            self.attitude_error_deg         = 22.0
            self.sun_vector_alignment       = math.cos(math.radians(22.0))
            self.reaction_wheel_momentum    = 0.40
            self.wheel_saturation_level     = 0.40
            self.gyro_bias                  = 1.5
            self.star_tracker_available     = True
            self.battery_temp               = 32.0
            self.payload_temp               = 68.0
            self.avionics_temp              = 58.0
            self.radiator_efficiency        = 0.55
            self.thermal_loop_health        = 0.65
            self.antenna_pointing_error     = 12.0
            self.transmitter_power          = 4.0
            self.bit_error_rate             = 0.018
            self.uplink_margin              = 7.5
            self.packet_loss_ratio          = 0.12
            self.command_latency_ms         = 320.0
            self.sunlit                     = True
            self.eclipse_timer              = 0
            self.ground_station_visible     = True
            self.radiation_zone             = False
            self.observation_window_active  = True    # active science window!
            self._orbit_step                = 3
            self.payload_on                 = True
            self.safe_mode                  = False
            self._faults                    = {_F_RADIATOR_STUCK, _F_AMPLIFIER_DEGRADE}
            self._fault_timers              = {_F_RADIATOR_STUCK: 0, _F_AMPLIFIER_DEGRADE: 0}

        else:  # hard
            # Cascading multi-subsystem failure: eclipse, GS blackout, radiation, 7 faults
            self.battery_soc                = 22.0
            self.bus_voltage                = 23.5
            self.panel_health               = 0.55
            self.charge_controller_health   = 0.50
            self.payload_power_draw         = 8.0
            self.attitude_error_deg         = 65.0
            self.sun_vector_alignment       = math.cos(math.radians(65.0))
            self.reaction_wheel_momentum    = 0.75
            self.wheel_saturation_level     = 0.75
            self.gyro_bias                  = 3.2
            self.star_tracker_available     = False
            self.battery_temp               = 3.0
            self.payload_temp               = 74.0
            self.avionics_temp              = 65.0
            self.radiator_efficiency        = 0.35
            self.thermal_loop_health        = 0.40
            self.antenna_pointing_error     = 35.0
            self.transmitter_power          = 2.8
            self.bit_error_rate             = 0.055
            self.uplink_margin              = 3.2
            self.packet_loss_ratio          = 0.38
            self.command_latency_ms         = 850.0
            self.sunlit                     = False
            self.eclipse_timer              = 4
            self.ground_station_visible     = False
            self.radiation_zone             = True
            self.observation_window_active  = False
            self._orbit_step                = 8
            self.payload_on                 = True
            self.safe_mode                  = False
            self._faults                    = {
                _F_RW_SATURATION, _F_GYRO_DRIFT, _F_STAR_TRACKER_DROP,
                _F_HEAT_PIPE_FAIL, _F_HEATER_LATCH, _F_TRANSPONDER_HOT,
                _F_MPPT_STUCK,
            }
            self._fault_timers = {f: 0 for f in self._faults}

    # ──────────────────────────────────────────────────────────────────
    # Orbital Context
    # ──────────────────────────────────────────────────────────────────

    # Deterministic 16-step orbital schedule: (sunlit, gs_visible, obs_window, radiation)
    _ORBIT_SCHEDULE = [
        (True,  True,  False, False),  # 0
        (True,  True,  False, False),  # 1
        (True,  False, True,  False),  # 2  science window begins
        (True,  False, True,  False),  # 3
        (True,  False, True,  False),  # 4
        (True,  True,  False, False),  # 5
        (True,  True,  False, False),  # 6
        (True,  False, False, True),   # 7  radiation belt
        (False, False, False, True),   # 8  eclipse + radiation
        (False, False, False, False),  # 9  eclipse
        (False, False, False, False),  # 10 eclipse
        (False, True,  False, False),  # 11 eclipse + GS contact
        (True,  True,  False, False),  # 12 eclipse exit
        (True,  True,  False, False),  # 13
        (True,  False, True,  False),  # 14 science window
        (True,  True,  False, False),  # 15
    ]

    def _advance_orbital_context(self) -> None:
        slot = self._orbit_step % len(self._ORBIT_SCHEDULE)
        sunlit, gs, obs, rad = self._ORBIT_SCHEDULE[slot]

        self.sunlit                    = sunlit
        self.ground_station_visible    = gs
        self.observation_window_active = obs
        self.radiation_zone            = rad

        if self.radiation_zone:
            self.panel_health = max(0.1, self.panel_health - 0.005)

        if not self.sunlit:
            self.eclipse_timer = self.eclipse_timer + 1
            self.heater_draw   = 4.0 if self.battery_temp < 5.0 else 2.0
            self.heater_state  = True
        else:
            self.eclipse_timer = 0
            self.heater_draw   = 0.5 if self.battery_temp < 0.0 else 0.0
            self.heater_state  = self.heater_draw > 0

        self._orbit_step += 1

    # ──────────────────────────────────────────────────────────────────
    # Action Application
    # ──────────────────────────────────────────────────────────────────

    def _apply_action(self, action_type: str) -> None:
        if action_type == "rotate_to_sun":
            wheel_authority = 1.0 - self.wheel_saturation_level
            if _F_RW_SATURATION in self._faults:
                wheel_authority *= 0.3
            reduction = 25.0 * wheel_authority
            self.attitude_error_deg   = max(2.0, self.attitude_error_deg - reduction)
            self.sun_vector_alignment = math.cos(math.radians(self.attitude_error_deg))
            self.reaction_wheel_momentum = max(0.0, self.reaction_wheel_momentum - 0.12)
            self.wheel_saturation_level  = max(0.0, self.wheel_saturation_level  - 0.12)
            self.antenna_pointing_error  = max(2.0, self.antenna_pointing_error  - 10.0)

        elif action_type == "disable_payload":
            self.payload_on         = False
            self.payload_power_draw = 0.0
            self.payload_temp       = max(self.payload_temp - 12.0, _TEMP_SPACE)

        elif action_type == "reboot_comms":
            if (_F_TRANSPONDER_HOT in self._faults
                    and self._fault_timers.get(_F_TRANSPONDER_HOT, 0) < 4):
                self._faults.discard(_F_TRANSPONDER_HOT)
            self.bit_error_rate     = max(0.001, self.bit_error_rate    * 0.5)
            self.packet_loss_ratio  = max(0.01,  self.packet_loss_ratio * 0.55)
            self.command_latency_ms = max(80.0,  self.command_latency_ms * 0.6)
            self.transmitter_power  = min(5.0,   self.transmitter_power + 0.8)

        elif action_type == "enter_safe_mode":
            self.safe_mode           = True
            self.payload_on          = False
            self.payload_power_draw  = 0.0
            self.attitude_error_deg  = max(5.0, self.attitude_error_deg - 8.0)
            self.sun_vector_alignment = math.cos(math.radians(self.attitude_error_deg))
            self.wheel_saturation_level = max(0.0, self.wheel_saturation_level - 0.10)
            self.payload_temp  = max(self.payload_temp  - 8.0,  _TEMP_SPACE)
            self.avionics_temp = max(self.avionics_temp - 4.0, _TEMP_SPACE)

        elif action_type == "switch_power_bus":
            self._faults.discard(_F_BUS_SHORT)
            self.power_bus_redundancy = True
            self.battery_soc = min(_SOC_FULL,   self.battery_soc + 6.0)
            self.bus_voltage = min(_BUS_NOMINAL, self.bus_voltage + 1.5)

        # "noop" → no change

    # ──────────────────────────────────────────────────────────────────
    # Fault Cascade Engine
    # ──────────────────────────────────────────────────────────────────

    def _tick_fault_cascades(self) -> None:
        for fault in list(self._faults):
            t = self._fault_timers.get(fault, 0) + 1
            self._fault_timers[fault] = t
            self._apply_fault_effect(fault, t)

    def _apply_fault_effect(self, fault: str, age: int) -> None:
        if fault == _F_MPPT_STUCK:
            self.charge_controller_health = max(0.2, 1.0 - age * 0.06)

        elif fault == _F_PANEL_JAM:
            self.panel_health = max(0.1, self.panel_health - 0.03)

        elif fault == _F_BUS_SHORT:
            self.bus_voltage = max(18.0, _BUS_NOMINAL - age * 0.8)
            if age >= 3:
                self.avionics_temp = min(90.0, self.avionics_temp + 3.0)

        elif fault == _F_BAT_AGING:
            self.battery_soc = max(0.0, self.battery_soc - 0.5)

        elif fault == _F_RW_SATURATION:
            self.reaction_wheel_momentum = min(1.0, self.reaction_wheel_momentum + 0.04)
            self.wheel_saturation_level  = min(1.0, self.wheel_saturation_level  + 0.04)
            if age >= 2:
                self.attitude_error_deg   = min(90.0, self.attitude_error_deg + 3.0)
                self.sun_vector_alignment = math.cos(math.radians(self.attitude_error_deg))

        elif fault == _F_GYRO_DRIFT:
            self.gyro_bias = min(10.0, self.gyro_bias + 0.4)
            if age >= 3:
                self.antenna_pointing_error = min(60.0, self.antenna_pointing_error + 2.0)

        elif fault == _F_STAR_TRACKER_DROP:
            self.star_tracker_available = False
            if age >= 2:
                self.attitude_error_deg   = min(90.0, self.attitude_error_deg + 2.5)
                self.sun_vector_alignment = math.cos(math.radians(self.attitude_error_deg))

        elif fault == _F_RADIATOR_STUCK:
            self.radiator_efficiency = max(0.05, 0.6 - age * 0.05)
            if age >= 2:
                self.avionics_temp = min(90.0, self.avionics_temp + 2.5)

        elif fault == _F_HEAT_PIPE_FAIL:
            self.thermal_loop_health = max(0.05, 1.0 - age * 0.08)
            if age >= 3:
                self.payload_temp = min(110.0, self.payload_temp + 3.0)

        elif fault == _F_HEATER_LATCH:
            self.heater_state = False
            self.heater_draw  = 0.0
            if not self.sunlit:
                self.battery_temp = max(-30.0, self.battery_temp - 3.0)

        elif fault == _F_TRANSPONDER_HOT:
            self.bit_error_rate    = min(0.5, self.bit_error_rate    + 0.015 * age)
            self.packet_loss_ratio = min(0.9, self.packet_loss_ratio + 0.012 * age)
            if age >= 4:
                self.avionics_temp = min(90.0, self.avionics_temp + 2.0)

        elif fault == _F_AMPLIFIER_DEGRADE:
            self.transmitter_power = max(0.5,  self.transmitter_power - 0.15)
            self.uplink_margin     = max(-5.0, self.uplink_margin     - 0.5)

        elif fault == _F_ANTENNA_STALL:
            self.antenna_pointing_error = min(60.0, self.antenna_pointing_error + 1.5)

    # ──────────────────────────────────────────────────────────────────
    # EPS Update
    # ──────────────────────────────────────────────────────────────────

    def _eps_update(self) -> None:
        if self.sunlit:
            raw_current = (
                self.sun_vector_alignment
                * self.panel_health
                * self.charge_controller_health
                * 5.5
            )
            if _F_MPPT_STUCK in self._faults:
                raw_current *= max(0.2, self.charge_controller_health)
            self.solar_array_current = max(0.0, raw_current)
        else:
            self.solar_array_current = 0.0

        p_gen  = self.solar_array_current * self.bus_voltage
        p_load = (
            (self.payload_power_draw if self.payload_on else 0.0)
            + self.avionics_draw
            + self.heater_draw
        )
        net_power = p_gen - p_load
        delta_soc = net_power * 0.4 * _DT
        self.battery_soc = max(_SOC_MIN, min(_SOC_FULL, self.battery_soc + delta_soc))

        if net_power < 0:
            self.bus_voltage = max(18.0, self.bus_voltage - abs(net_power) * 0.015)
        else:
            self.bus_voltage = min(_BUS_NOMINAL, self.bus_voltage + net_power * 0.005)

        if _F_BAT_AGING in self._faults:
            self.battery_soc = max(0.0, self.battery_soc - 0.8)

        if self.safe_mode:
            self.payload_power_draw = 0.0

    # ──────────────────────────────────────────────────────────────────
    # ADCS Update
    # ──────────────────────────────────────────────────────────────────

    def _adcs_update(self) -> None:
        drift_rate = 1.5
        if _F_GYRO_DRIFT in self._faults:
            drift_rate += self.gyro_bias * 0.3
        self.attitude_error_deg   = min(90.0, self.attitude_error_deg + drift_rate)
        self.sun_vector_alignment = math.cos(math.radians(self.attitude_error_deg))

        self.reaction_wheel_momentum = min(1.0, self.reaction_wheel_momentum + 0.02)
        self.wheel_saturation_level  = min(1.0, self.wheel_saturation_level  + 0.02)
        self.gyro_bias = min(10.0, self.gyro_bias + 0.05)

        self.antenna_pointing_error = min(60.0,
            3.0 + self.attitude_error_deg * 0.5 + self.gyro_bias * 0.3
        )

    # ──────────────────────────────────────────────────────────────────
    # Thermal Update
    # ──────────────────────────────────────────────────────────────────

    def _thermal_update(self) -> None:
        if self.payload_on:
            self.payload_temp += 5.0
        else:
            self.payload_temp -= 2.0

        radiator_cooling = 4.0 * self.radiator_efficiency * self.thermal_loop_health
        self.payload_temp  -= radiator_cooling * 0.6
        self.avionics_temp -= radiator_cooling * 0.3
        self.battery_temp  -= radiator_cooling * 0.1

        conduction_pa = (self.payload_temp  - self.avionics_temp) * 0.08
        self.avionics_temp += conduction_pa
        conduction_ab = (self.avionics_temp - self.battery_temp)  * 0.06
        self.battery_temp  += conduction_ab

        for attr, k in [("payload_temp", 0.04), ("avionics_temp", 0.05), ("battery_temp", 0.06)]:
            t = getattr(self, attr)
            setattr(self, attr, t - k * (t - _TEMP_SPACE))

        if not self.sunlit:
            self.battery_temp -= 2.5
            if self.heater_state and _F_HEATER_LATCH not in self._faults:
                self.battery_temp += min(3.5, self.heater_draw * 0.6)

        if self.safe_mode:
            self.payload_temp  = max(self.payload_temp  - 3.0, _TEMP_SPACE)
            self.avionics_temp = max(self.avionics_temp - 2.0, _TEMP_SPACE)

        self.payload_temp  = max(-40.0, min(120.0, self.payload_temp))
        self.avionics_temp = max(-40.0, min(100.0, self.avionics_temp))
        self.battery_temp  = max(-40.0, min( 60.0, self.battery_temp))

    # ──────────────────────────────────────────────────────────────────
    # Communications Update
    # ──────────────────────────────────────────────────────────────────

    def _comms_update(self) -> None:
        pointing_loss    = self.antenna_pointing_error * 0.25
        effective_margin = self.uplink_margin - pointing_loss

        if self.avionics_temp > 65:
            self.bit_error_rate = min(0.9,
                self.bit_error_rate + (self.avionics_temp - 65) * 0.003)

        if effective_margin < 5.0:
            deficit = 5.0 - effective_margin
            self.bit_error_rate    = min(0.9,  self.bit_error_rate    + deficit * 0.004)
            self.packet_loss_ratio = min(0.95, self.packet_loss_ratio + deficit * 0.003)

        self.command_latency_ms = max(80.0, 120.0 + self.packet_loss_ratio * 1500.0)
        if not self.ground_station_visible:
            self.command_latency_ms = min(self.command_latency_ms + 200.0, 9999.0)

        self.bit_error_rate    = max(0.0001, min(0.99, self.bit_error_rate))
        self.packet_loss_ratio = max(0.001,  min(0.99, self.packet_loss_ratio))
        self.uplink_margin     = max(-20.0,  min(20.0, self.uplink_margin))

    # ──────────────────────────────────────────────────────────────────
    # Done Condition
    # ──────────────────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        stabilised = (
            self.battery_soc         >= 60.0
            and self.payload_temp    <= 55.0
            and self.avionics_temp   <= 55.0
            and self.battery_temp    >= 0.0
            and self.attitude_error_deg <= 15.0
            and self.bit_error_rate  <= 0.01
        )
        return stabilised or self._state.step_count >= 12

    # ──────────────────────────────────────────────────────────────────
    # Multi-Objective Reward
    # ──────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """Dense multi-objective mission utility in strict (0, 1)."""
        soc_score     = self.battery_soc / 100.0
        voltage_score = max(0.0, (self.bus_voltage - 18.0) / (_BUS_NOMINAL - 18.0))
        eps_score     = 0.6 * soc_score + 0.4 * voltage_score

        payload_thermal  = max(0.0, 1.0 - max(0.0, self.payload_temp  - 25.0) / 80.0)
        avionics_thermal = max(0.0, 1.0 - max(0.0, self.avionics_temp - 20.0) / 65.0)
        battery_thermal  = max(0.0, 1.0 - abs(self.battery_temp - 15.0)       / 45.0)
        thermal_score    = (payload_thermal + avionics_thermal + battery_thermal) / 3.0

        attitude_score = max(0.0, 1.0 - self.attitude_error_deg / 90.0)
        wheel_score    = max(0.0, 1.0 - self.wheel_saturation_level)
        adcs_score     = 0.7 * attitude_score + 0.3 * wheel_score

        ber_score   = max(0.0, 1.0 - self.bit_error_rate    / 0.1)
        plr_score   = max(0.0, 1.0 - self.packet_loss_ratio / 0.5)
        comms_score = 0.5 * ber_score + 0.5 * plr_score

        science_bonus = 0.0
        if self.observation_window_active and self.payload_on and not self.safe_mode:
            science_bonus = 0.12

        survivability = 1.0
        if self.battery_soc   < 10.0:  survivability *= 0.4
        if self.battery_temp  < -10.0: survivability *= 0.6
        if self.avionics_temp > 80.0:  survivability *= 0.5
        if self.payload_temp  > 90.0:  survivability *= 0.6

        raw = (
            0.30 * eps_score
            + 0.22 * thermal_score
            + 0.18 * adcs_score
            + 0.15 * comms_score
            + 0.15 * survivability
        ) * survivability + science_bonus

        return self._safe_reward(min(1.0, max(0.0, raw)))

    @staticmethod
    def _safe_reward(raw: float) -> float:
        """Map any float to strict open interval (0.001, 0.999)."""
        eps = 0.001
        clamped = max(0.0, min(1.0, raw))
        return round(eps + clamped * (1.0 - 2 * eps), 4)

    # ──────────────────────────────────────────────────────────────────
    # Partial Observability
    # ──────────────────────────────────────────────────────────────────

    def _get_dropout_fields(self) -> set:
        """Return deterministic set of field names to hide this step."""
        step    = self._state.step_count
        dropped = set()

        if not self.star_tracker_available or _F_STAR_TRACKER_DROP in self._faults:
            dropped.add("star_tracker_available")
        if _F_GYRO_DRIFT in self._faults and step % 4 == 0:
            dropped.add("gyro_bias")
        if not self.ground_station_visible:
            dropped.add("uplink_margin")
            dropped.add("command_latency_ms")
        if step % 3 == 2:
            dropped.add("avionics_temp")
        if step % 5 == 0:
            dropped.add("solar_array_current")

        return dropped

    # ──────────────────────────────────────────────────────────────────
    # Mission Status
    # ──────────────────────────────────────────────────────────────────

    def _mission_status(self) -> str:
        if (self.battery_soc   < 15.0
                or self.battery_temp  < -15.0
                or self.avionics_temp > 80.0
                or self.payload_temp  > 85.0):
            return "critical"
        if (self.battery_soc   < 35.0
                or self.attitude_error_deg > 45.0
                or self.payload_temp  > 65.0
                or self.avionics_temp > 65.0):
            return "warning"
        return "stable"

    # ──────────────────────────────────────────────────────────────────
    # Observation Construction
    # ──────────────────────────────────────────────────────────────────

    def _get_observation(self, reward: float, done: bool) -> OrbitalAnomalyOpenenvObservation:
        dropped = self._get_dropout_fields()

        def maybe(field, value, default=None):
            return default if field in dropped else value

        return OrbitalAnomalyOpenenvObservation(
            # V1 backward-compatible fields
            battery_level=round(self.battery_soc, 2),
            solar_efficiency=round(max(0.0, self.sun_vector_alignment * self.panel_health), 4),
            thermal_temp=round(self.payload_temp, 2),
            comms_signal=round(max(0.0, min(1.0,
                1.0 - self.bit_error_rate * 5.0 - self.packet_loss_ratio)), 4),
            payload_on=self.payload_on,
            safe_mode=self.safe_mode,
            task_id=self.task_id,
            mission_status=self._mission_status(),
            reward=reward,
            done=done,

            # V2 EPS
            battery_soc=round(self.battery_soc, 2),
            bus_voltage=round(self.bus_voltage, 3),
            panel_health=round(self.panel_health, 4),
            solar_array_current=round(
                maybe("solar_array_current", self.solar_array_current, -1.0), 3),
            charge_controller_health=round(self.charge_controller_health, 4),
            power_bus_redundancy=self.power_bus_redundancy,

            # V2 ADCS
            attitude_error_deg=round(self.attitude_error_deg, 2),
            sun_vector_alignment=round(self.sun_vector_alignment, 4),
            reaction_wheel_momentum=round(self.reaction_wheel_momentum, 4),
            gyro_bias=round(maybe("gyro_bias", self.gyro_bias, -999.0), 3),
            star_tracker_available=maybe("star_tracker_available", self.star_tracker_available, None),
            wheel_saturation_level=round(self.wheel_saturation_level, 4),

            # V2 Thermal
            battery_temp=round(self.battery_temp, 2),
            payload_temp=round(self.payload_temp, 2),
            avionics_temp=round(
                maybe("avionics_temp", self.avionics_temp, -999.0), 2),
            radiator_efficiency=round(self.radiator_efficiency, 4),
            heater_state=self.heater_state,
            thermal_loop_health=round(self.thermal_loop_health, 4),

            # V2 Comms
            antenna_pointing_error=round(self.antenna_pointing_error, 2),
            transmitter_power=round(self.transmitter_power, 3),
            bit_error_rate=round(self.bit_error_rate, 5),
            uplink_margin=round(
                maybe("uplink_margin", self.uplink_margin, -99.0), 2),
            packet_loss_ratio=round(self.packet_loss_ratio, 4),
            command_latency_ms=round(
                maybe("command_latency_ms", self.command_latency_ms, -1.0), 1),

            # V2 Orbital context
            sunlit=self.sunlit,
            eclipse_timer=self.eclipse_timer,
            ground_station_visible=self.ground_station_visible,
            radiation_zone=self.radiation_zone,
            observation_window_active=self.observation_window_active,

            metadata={
                "step": self._state.step_count,
                "episode_id": self._state.episode_id,
                "version": "2.0",
                "obs_dropout": list(dropped),
            },
        )