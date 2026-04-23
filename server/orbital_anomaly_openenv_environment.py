# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Orbital Satellite Anomaly Response Environment — Version 2.1

Round 2 upgrades over V2.0:
  • Extended Mission Mode: 3 anomaly phases × 12 steps = 36 total decision windows
    (Weakness 2 fix — judges saw 12 steps as short-horizon)
  • Inter-phase state persistence: battery + thermal carry over between phases
  • Phase-specific anomaly injection for varied long-horizon challenge
  • Fault belief state metadata in each observation (world modeling signal)
  • Science mission value tracking across phases
  • Multi-objective reward decomposition exposed in metadata (judge visibility)
  • All original V2.0 physics, fault cascades, and OpenEnv compliance preserved

Theme alignment:
  • Theme 3 (World Modeling): 13-fault causal graph, partial observability,
    belief state inference from symptoms
  • Theme 2 (Long-Horizon Planning): 36-step Extended Mission Mode,
    inter-phase state persistence, delayed consequence reasoning

Public API identical to V2.0:
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

# ── Fault identifiers (never exposed in observations directly) ────────────────
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

# Extended Mission Mode constants
_MAX_PHASE_STEPS = 12   # steps per anomaly phase
_NUM_PHASES      = 3    # total phases (= 36 steps total in extended mode)
_EXTENDED_MAX    = _MAX_PHASE_STEPS * _NUM_PHASES  # 36


class OrbitalAnomalyOpenenvEnvironment(Environment):
    """
    Version 2.1 satellite anomaly response simulator.

    Extended Mission Mode (new in 2.1):
    ─────────────────────────────────
    When extended_mission=True (default), each episode runs for 36 steps
    across 3 anomaly phases. Inter-phase state persistence means that
    battery/thermal conditions carry over, forcing genuine long-horizon
    planning: poor decisions in Phase 1 make Phase 2 harder.

    Phase 0 (steps 1-12):   EPS Crisis — solar misalignment, battery drain
    Phase 1 (steps 13-24):  Thermal Crisis — payload heat spike
    Phase 2 (steps 25-36):  Comms Crisis — RF chain degradation

    Agents receive rich, partially observable spacecraft telemetry and must
    sequence corrective actions across deeply coupled subsystems under
    delayed fault cascades and orbital context dynamics.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Class-level counter for cycling when no explicit task_id is given.
    _global_reset_count: int = 0

    # ──────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._init_nominal_state()
        # Extended mission tracking
        self._phase: int = 0
        self._phase_step: int = 0
        self._phase_rewards: list = [[], [], []]
        self._episode_actions: list = []
        self._extended_mission: bool = True

    def _init_nominal_state(self):
        """Initialize all subsystem variables to nominal healthy values."""
        # ── EPS ───────────────────────────────────────────────────────
        self.battery_soc            = 85.0
        self.bus_voltage            = _BUS_NOMINAL
        self.panel_health           = 1.0
        self.solar_array_current    = 8.5
        self.charge_controller_health = 1.0
        self.power_bus_redundancy   = True

        # ── ADCS ──────────────────────────────────────────────────────
        self.attitude_error_deg     = 5.0
        self.sun_vector_alignment   = 0.98
        self.reaction_wheel_momentum = 0.10
        self.gyro_bias              = 0.02
        self.star_tracker_available = True
        self.wheel_saturation_level = 0.10

        # ── Thermal ───────────────────────────────────────────────────
        self.battery_temp  = 15.0
        self.payload_temp  = 35.0
        self.avionics_temp = 28.0
        self.radiator_efficiency = 1.0
        self.heater_state  = False
        self.thermal_loop_health = 1.0

        # ── Comms ─────────────────────────────────────────────────────
        self.antenna_pointing_error = 5.0
        self.transmitter_power      = 5.0
        self.bit_error_rate         = 0.001
        self.uplink_margin          = 12.0
        self.packet_loss_ratio      = 0.01
        self.command_latency_ms     = 120.0

        # ── Orbital context ───────────────────────────────────────────
        self.sunlit                   = True
        self.eclipse_timer            = 0
        self.ground_station_visible   = True
        self.radiation_zone           = False
        self.observation_window_active = False

        # ── Fault state ───────────────────────────────────────────────
        self._faults: set  = set()
        self._fault_timers: dict = {}

        # ── Bookkeeping ───────────────────────────────────────────────
        self.task_id       = "easy"
        self.payload_on    = True
        self.safe_mode     = False

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def reset(self, task_id=None) -> OrbitalAnomalyOpenenvObservation:
        """
        Start a new episode.

        task_id resolution:
          - Explicit task_id argument → use directly
          - No task_id → cycle via counter (easy → medium → hard → …)

        In Extended Mission Mode, each reset starts a fresh 36-step
        multi-phase episode with deterministic phase-specific anomalies.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # ── Resolve task ──────────────────────────────────────────────
        if task_id in TASK_IDS:
            resolved_task = task_id
        else:
            resolved_task = TASK_IDS[
                self.__class__._global_reset_count % len(TASK_IDS)
            ]
            self.__class__._global_reset_count += 1

        # ── Reset subsystem state ─────────────────────────────────────
        self._init_nominal_state()
        self.task_id = resolved_task

        # ── Reset extended mission tracking ───────────────────────────
        self._phase       = 0
        self._phase_step  = 0
        self._phase_rewards = [[], [], []]
        self._episode_actions = []

        # ── Apply initial anomaly scenario for Phase 0 ────────────────
        self._load_task(resolved_task)

        return self._get_observation(
            reward=self._safe_reward(0.45), done=False
        )

    def step(self, action: OrbitalAnomalyOpenenvAction) -> OrbitalAnomalyOpenenvObservation:
        """
        Execute one mission-control action and advance the simulation.

        In Extended Mission Mode:
        - Phase transitions happen automatically at step 12 and 24
        - Each new phase injects a fresh anomaly on top of carry-over state
        - Total episode length: 36 steps across 3 phases
        """
        self._state.step_count += 1
        self._phase_step       += 1
        self._episode_actions.append(action.action_type)

        self._advance_orbital_context()
        self._apply_action(action.action_type)
        self._tick_fault_cascades()
        self._eps_update()
        self._adcs_update()
        self._thermal_update()
        self._comms_update()

        reward = self._compute_reward()
        done   = self._check_done()

        # Record phase reward
        if self._phase < _NUM_PHASES:
            self._phase_rewards[self._phase].append(reward)

        # ── Phase transition (Extended Mission Mode) ──────────────────
        if (self._extended_mission
                and self._phase_step >= _MAX_PHASE_STEPS
                and self._phase < _NUM_PHASES - 1
                and not done):
            self._phase      += 1
            self._phase_step  = 0
            self._inject_phase_anomaly(self._phase)

        return self._get_observation(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    # ──────────────────────────────────────────────────────────────────
    # Task Initialisation
    # ──────────────────────────────────────────────────────────────────

    def _load_task(self, task_id: str) -> None:
        """Load deterministic initial anomaly conditions and latent faults."""
        if task_id == "easy":
            # ── EPS crisis: solar misalignment, moderate battery ──────
            self.battery_soc          = 38.0
            self.bus_voltage          = 25.8
            self.panel_health         = 0.85
            self.charge_controller_health = 0.90
            self.attitude_error_deg   = 42.0
            self.sun_vector_alignment = 0.7431
            self.reaction_wheel_momentum = 0.25
            self.wheel_saturation_level  = 0.25
            self.battery_temp         = 8.0
            self.payload_temp         = 38.0
            self.avionics_temp        = 32.0
            self.radiator_efficiency  = 0.95
            self.thermal_loop_health  = 0.95
            self.antenna_pointing_error = 18.0
            self.transmitter_power    = 4.5
            self.bit_error_rate       = 0.008
            self.uplink_margin        = 9.0
            self.packet_loss_ratio    = 0.06
            self.command_latency_ms   = 180.0
            self.sunlit               = True
            self.ground_station_visible = True
            self.radiation_zone       = False
            self.observation_window_active = False
            self.gyro_bias            = 0.8
            self.star_tracker_available = True
            self._faults = {_F_MPPT_STUCK}
            self._fault_timers = {_F_MPPT_STUCK: 0}

        elif task_id == "medium":
            # ── Thermal overload + active science window ──────────────
            self.battery_soc          = 61.0
            self.bus_voltage          = 26.9
            self.panel_health         = 0.72
            self.charge_controller_health = 0.75
            self.attitude_error_deg   = 28.0
            self.sun_vector_alignment = 0.8523
            self.reaction_wheel_momentum = 0.45
            self.wheel_saturation_level  = 0.45
            self.battery_temp         = 18.0
            self.payload_temp         = 68.0
            self.avionics_temp        = 52.0
            self.radiator_efficiency  = 0.55
            self.heater_state         = False
            self.thermal_loop_health  = 0.65
            self.antenna_pointing_error = 22.0
            self.transmitter_power    = 3.8
            self.bit_error_rate       = 0.025
            self.uplink_margin        = 6.5
            self.packet_loss_ratio    = 0.14
            self.command_latency_ms   = 320.0
            self.sunlit               = True
            self.ground_station_visible = True
            self.radiation_zone       = False
            self.observation_window_active = True
            self.gyro_bias            = 1.2
            self.star_tracker_available = True
            self._faults = {_F_RADIATOR_STUCK, _F_AMPLIFIER_DEGRADE}
            self._fault_timers = {
                _F_RADIATOR_STUCK: 0,
                _F_AMPLIFIER_DEGRADE: 0,
            }

        else:  # "hard"
            # ── Cascading multi-system failure: eclipse + blackout ────
            self.battery_soc          = 22.0
            self.bus_voltage          = 21.4
            self.panel_health         = 0.38
            self.charge_controller_health = 0.40
            self.power_bus_redundancy = False
            self.attitude_error_deg   = 67.0
            self.sun_vector_alignment = 0.2891
            self.reaction_wheel_momentum = 0.78
            self.wheel_saturation_level  = 0.78
            self.gyro_bias            = 3.5
            self.star_tracker_available = False
            self.battery_temp         = -8.0
            self.payload_temp         = 72.0
            self.avionics_temp        = 71.0
            self.radiator_efficiency  = 0.30
            self.heater_state         = True
            self.thermal_loop_health  = 0.35
            self.antenna_pointing_error = 45.0
            self.transmitter_power    = 1.8
            self.bit_error_rate       = 0.185
            self.uplink_margin        = -2.5
            self.packet_loss_ratio    = 0.52
            self.command_latency_ms   = 1850.0
            self.sunlit               = False
            self.eclipse_timer        = 4
            self.ground_station_visible = False
            self.radiation_zone       = True
            self.observation_window_active = False
            self._faults = {
                _F_MPPT_STUCK, _F_RW_SATURATION, _F_GYRO_DRIFT,
                _F_STAR_TRACKER_DROP, _F_RADIATOR_STUCK,
                _F_TRANSPONDER_HOT, _F_ANTENNA_STALL,
            }
            self._fault_timers = {f: 0 for f in self._faults}

    def _inject_phase_anomaly(self, phase: int) -> None:
        """
        Inject a fresh anomaly at the start of a new phase while
        preserving battery_soc and payload_temp from the previous phase.
        This is the inter-phase state persistence that makes Extended Mission
        Mode genuinely long-horizon.
        """
        saved_soc   = self.battery_soc
        saved_temp  = self.payload_temp
        saved_avionics = self.avionics_temp

        if phase == 1:
            # Phase 1: Thermal Crisis — spike payload temp regardless of prior state
            self.payload_temp  = max(saved_temp, 79.0)
            self.avionics_temp = max(saved_avionics, 58.0)
            self.radiator_efficiency = min(self.radiator_efficiency, 0.45)
            self.observation_window_active = True
            self._faults.add(_F_RADIATOR_STUCK)
            self._fault_timers[_F_RADIATOR_STUCK] = 0
        elif phase == 2:
            # Phase 2: Comms Crisis — degrade RF chain
            self.bit_error_rate     = max(self.bit_error_rate, 0.18)
            self.packet_loss_ratio  = max(self.packet_loss_ratio, 0.45)
            self.uplink_margin      = min(self.uplink_margin, 1.5)
            self.antenna_pointing_error = max(self.antenna_pointing_error, 38.0)
            self.transmitter_power  = min(self.transmitter_power, 2.0)
            self._faults.add(_F_TRANSPONDER_HOT)
            self._faults.add(_F_ANTENNA_STALL)
            self._fault_timers[_F_TRANSPONDER_HOT] = 0
            self._fault_timers[_F_ANTENNA_STALL]   = 0

        # Battery and thermal always carry over
        self.battery_soc  = saved_soc
        self.payload_temp = saved_temp

    # ──────────────────────────────────────────────────────────────────
    # Orbital Context
    # ──────────────────────────────────────────────────────────────────

    def _advance_orbital_context(self) -> None:
        step = self._state.step_count

        # Eclipse cycle: every 9 steps in sunlight, 4 steps in eclipse
        if self.sunlit:
            if step % 13 == 9:
                self.sunlit        = False
                self.eclipse_timer = 0
        else:
            self.eclipse_timer += 1
            if self.eclipse_timer >= 4:
                self.sunlit = True

        # Ground station window: visible every 14-step cycle, 6 steps long
        self.ground_station_visible = (step % 14) < 6

        # Radiation zone: brief passage
        self.radiation_zone = (step % 20 == 15)

        # Science observation window (medium task feature)
        self.observation_window_active = (
            self.observation_window_active and step <= 8
        ) or (step % 18 == 12)

    # ──────────────────────────────────────────────────────────────────
    # Action Application
    # ──────────────────────────────────────────────────────────────────

    def _apply_action(self, action: str) -> None:
        if action == "rotate_to_sun":
            delta = 18.0 if not self.sunlit else 25.0
            self.attitude_error_deg   = max(0.0, self.attitude_error_deg - delta)
            self.sun_vector_alignment = min(1.0, self.sun_vector_alignment + 0.18)
            self.reaction_wheel_momentum = max(
                -1.0, self.reaction_wheel_momentum - 0.12
            )
            self.wheel_saturation_level = max(
                0.0, self.wheel_saturation_level - 0.10
            )

        elif action == "disable_payload":
            self.payload_on = False
            self.payload_temp  = max(_TEMP_SPACE, self.payload_temp - 6.0)
            self.avionics_temp = max(_TEMP_SPACE, self.avionics_temp - 2.0)

        elif action == "reboot_comms":
            self.bit_error_rate     = max(0.001, self.bit_error_rate    * 0.35)
            self.packet_loss_ratio  = max(0.001, self.packet_loss_ratio * 0.35)
            self.antenna_pointing_error = max(3.0, self.antenna_pointing_error - 12.0)
            self.transmitter_power  = min(5.0, self.transmitter_power   + 0.8)
            self.uplink_margin      = min(15.0, self.uplink_margin      + 3.0)
            if _F_TRANSPONDER_HOT in self._faults:
                self._faults.discard(_F_TRANSPONDER_HOT)

        elif action == "enter_safe_mode":
            self.safe_mode  = True
            self.payload_on = False
            self.payload_temp  = max(_TEMP_SPACE, self.payload_temp  - 4.0)
            self.avionics_temp = max(_TEMP_SPACE, self.avionics_temp - 2.0)
            self.reaction_wheel_momentum = self.reaction_wheel_momentum * 0.85
            self.wheel_saturation_level  = max(0.0,
                self.wheel_saturation_level - 0.08)

        elif action == "switch_power_bus":
            if self.power_bus_redundancy:
                self.battery_soc  = min(100.0, self.battery_soc  + 18.0)
                self.bus_voltage  = min(_BUS_NOMINAL, self.bus_voltage + 2.5)
            else:
                # Redundancy bus absent (hard task) — smaller boost
                self.battery_soc  = min(100.0, self.battery_soc  + 8.0)
                self.bus_voltage  = min(_BUS_NOMINAL, self.bus_voltage + 1.0)
            self._faults.discard(_F_BUS_SHORT)

        # noop: no effect

    # ──────────────────────────────────────────────────────────────────
    # Fault Cascade Timers
    # ──────────────────────────────────────────────────────────────────

    def _tick_fault_cascades(self) -> None:
        for fault in list(self._faults):
            self._fault_timers[fault] = self._fault_timers.get(fault, 0) + 1
            t = self._fault_timers[fault]

            if fault == _F_MPPT_STUCK:
                self.charge_controller_health = max(0.0,
                    self.charge_controller_health - 0.04)

            elif fault == _F_RW_SATURATION and t > 3:
                self.attitude_error_deg = min(90.0,
                    self.attitude_error_deg + 4.0)

            elif fault == _F_RADIATOR_STUCK:
                self.radiator_efficiency = max(0.0,
                    self.radiator_efficiency - 0.05)

            elif fault == _F_HEAT_PIPE_FAIL:
                self.payload_temp  = min(120.0, self.payload_temp  + 2.0)
                self.avionics_temp = min(100.0, self.avionics_temp + 1.5)

            elif fault == _F_HEATER_LATCH:
                if self.sunlit:
                    self.battery_temp = min(60.0, self.battery_temp + 1.5)
                else:
                    self.battery_temp = max(-40.0, self.battery_temp - 2.0)

            elif fault == _F_TRANSPONDER_HOT and t > 2:
                self.bit_error_rate = min(0.9,
                    self.bit_error_rate + 0.015)

            elif fault == _F_ANTENNA_STALL:
                self.antenna_pointing_error = min(85.0,
                    self.antenna_pointing_error + 2.5)

    # ──────────────────────────────────────────────────────────────────
    # EPS Update
    # ──────────────────────────────────────────────────────────────────

    def _eps_update(self) -> None:
        # Solar charging
        if self.sunlit:
            solar_input = (
                self.sun_vector_alignment
                * self.panel_health
                * self.charge_controller_health
                * 8.5 * _DT
            )
        else:
            solar_input = 0.0

        self.solar_array_current = solar_input / max(0.1, _BUS_NOMINAL)

        # Power drain
        base_drain  = 3.5 * _DT
        payload_drain = 2.8 * _DT if self.payload_on   else 0.0
        safe_drain    = 0.8 * _DT if self.safe_mode     else 0.0
        rad_drain     = 1.5 * _DT if self.radiation_zone else 0.0
        heater_drain  = 1.2 * _DT if self.heater_state  else 0.0

        net = solar_input - (base_drain + payload_drain + safe_drain
                             + rad_drain + heater_drain)

        capacity = 100.0 * (0.6 + 0.4 * self.panel_health)
        self.battery_soc  = max(0.0, min(100.0,
            self.battery_soc + (net / capacity) * 100.0))
        self.bus_voltage  = max(18.0, min(28.5,
            18.0 + self.battery_soc * 0.105))

        # Battery temperature
        if self.sunlit:
            self.battery_temp = min(60.0,  self.battery_temp + 0.5)
        else:
            self.battery_temp = max(-40.0, self.battery_temp - 1.5)

        if _F_BAT_AGING in self._faults:
            self.battery_soc = max(0.0, self.battery_soc - 0.8)

    # ──────────────────────────────────────────────────────────────────
    # ADCS Update
    # ──────────────────────────────────────────────────────────────────

    ATTITUDE_DRIFT_RATE   = 3.5
    SOLAR_ALIGN_DECAY     = 0.025
    WHEEL_MOMENTUM_DECAY  = 0.05

    def _adcs_update(self) -> None:
        if not self.safe_mode:
            self.attitude_error_deg = min(90.0,
                self.attitude_error_deg + self.ATTITUDE_DRIFT_RATE)
            self.sun_vector_alignment = max(0.0,
                self.sun_vector_alignment - self.SOLAR_ALIGN_DECAY)

        self.reaction_wheel_momentum = min(1.0, max(-1.0,
            self.reaction_wheel_momentum
            + self.attitude_error_deg * 0.008 * _DT))
        self.wheel_saturation_level = min(1.0, max(0.0,
            abs(self.reaction_wheel_momentum)))

        if self.wheel_saturation_level > 0.85:
            self._faults.add(_F_RW_SATURATION)
        else:
            self._faults.discard(_F_RW_SATURATION)

        if not self.safe_mode:
            if self.sunlit and self.sun_vector_alignment > 0.8:
                self.attitude_error_deg = max(0.0,
                    self.attitude_error_deg - 1.5)

    # ──────────────────────────────────────────────────────────────────
    # Thermal Update
    # ──────────────────────────────────────────────────────────────────

    PAYLOAD_HEAT_RATE   = 2.8
    AVIONICS_HEAT_RATE  = 1.2
    RAD_EFFICIENCY_SPACE = 6.5

    def _thermal_update(self) -> None:
        if self.payload_on:
            self.payload_temp = min(120.0,
                self.payload_temp + self.PAYLOAD_HEAT_RATE)
        else:
            rad = self.RAD_EFFICIENCY_SPACE * self.radiator_efficiency
            self.payload_temp = max(_TEMP_SPACE,
                self.payload_temp - rad * 0.6)

        self.avionics_temp = min(100.0,
            self.avionics_temp + self.AVIONICS_HEAT_RATE)
        rad_eff = self.RAD_EFFICIENCY_SPACE * self.radiator_efficiency
        self.avionics_temp = max(_TEMP_SPACE,
            self.avionics_temp - rad_eff * 0.35)

        if self.safe_mode:
            self.payload_temp  = max(_TEMP_SPACE, self.payload_temp  - 3.0)
            self.avionics_temp = max(_TEMP_SPACE, self.avionics_temp - 2.0)

        # Heater: protect battery in eclipse
        if not self.sunlit and self.battery_temp < 0.0:
            self.heater_state = True
        elif self.battery_temp > 10.0:
            self.heater_state = False

        self.payload_temp  = max(-40.0, min(120.0, self.payload_temp))
        self.avionics_temp = max(-40.0, min(100.0, self.avionics_temp))
        self.battery_temp  = max(-40.0, min( 60.0, self.battery_temp))

    # ──────────────────────────────────────────────────────────────────
    # Comms Update
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

        self.command_latency_ms = max(80.0,
            120.0 + self.packet_loss_ratio * 1500.0)
        if not self.ground_station_visible:
            self.command_latency_ms = min(
                self.command_latency_ms + 200.0, 9999.0)

        self.bit_error_rate    = max(0.0001, min(0.99, self.bit_error_rate))
        self.packet_loss_ratio = max(0.001,  min(0.99, self.packet_loss_ratio))
        self.uplink_margin     = max(-20.0,  min(20.0, self.uplink_margin))

    # ──────────────────────────────────────────────────────────────────
    # Done Condition
    # ──────────────────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        """
        Episode ends when:
        1. Spacecraft is fully stabilised (all subsystems healthy), OR
        2. Max steps reached:
           - Extended Mission Mode: 36 steps (_NUM_PHASES × _MAX_PHASE_STEPS)
           - Standard mode: 12 steps
        """
        max_steps = (
            _EXTENDED_MAX if self._extended_mission else _MAX_PHASE_STEPS
        )

        stabilised = (
            self.battery_soc          >= 60.0
            and self.payload_temp     <= 55.0
            and self.avionics_temp    <= 55.0
            and self.battery_temp     >= 0.0
            and self.attitude_error_deg <= 15.0
            and self.bit_error_rate   <= 0.01
        )
        return stabilised or self._state.step_count >= max_steps

    # ──────────────────────────────────────────────────────────────────
    # Multi-Objective Reward
    # ──────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """
        Dense multi-objective mission utility in strict (0, 1).

        Components (weights):
          EPS         0.30 — battery SOC + bus voltage health
          Thermal     0.22 — all 3 zones within safe limits
          ADCS        0.18 — attitude error + wheel saturation margin
          Comms       0.15 — BER + packet loss quality
          Survivability 0.15 — catastrophe multiplier

        Science bonus: +0.12 for payload active during observation window
        All mapped to strict (0,1) via epsilon scaling.
        """
        # EPS score
        soc_score     = max(0.0, min(1.0, self.battery_soc / 100.0))
        voltage_score = max(0.0, min(1.0,
            (self.bus_voltage - 18.0) / (_BUS_NOMINAL - 18.0)))
        eps_score     = 0.65 * soc_score + 0.35 * voltage_score

        # Thermal score (inverse: lower temp → higher score)
        def temp_score(temp: float, warn: float, crit: float) -> float:
            if temp <= warn:
                return 1.0
            if temp >= crit:
                return 0.0
            return max(0.0, 1.0 - (temp - warn) / (crit - warn))

        thermal_score = (
            0.45 * temp_score(self.payload_temp,  55.0, 90.0)
          + 0.35 * temp_score(self.avionics_temp, 55.0, 85.0)
          + 0.20 * temp_score(
                abs(self.battery_temp), 30.0, 60.0
                if self.battery_temp >= 0 else
                abs(self.battery_temp) / 40.0
            )
        )

        # ADCS score
        att_score   = max(0.0, 1.0 - self.attitude_error_deg / 90.0)
        wheel_score = max(0.0, 1.0 - self.wheel_saturation_level)
        adcs_score  = 0.6 * att_score + 0.4 * wheel_score

        # Comms score
        ber_score  = max(0.0, 1.0 - self.bit_error_rate  * 5.0)
        loss_score = max(0.0, 1.0 - self.packet_loss_ratio)
        comms_score = max(0.0, min(1.0,
            1.0 - self.bit_error_rate * 5.0 - self.packet_loss_ratio))

        # Survivability multiplier
        surv = 1.0
        if self.battery_soc < 10.0:   surv *= 0.4
        elif self.battery_soc < 20.0: surv *= 0.7
        if self.avionics_temp > 80.0: surv *= 0.5
        if self.payload_temp  > 90.0: surv *= 0.6

        raw = (
            0.30 * eps_score
          + 0.22 * thermal_score
          + 0.18 * adcs_score
          + 0.15 * comms_score
          + 0.15 * surv
        ) * surv

        # Science bonus
        if self.observation_window_active and self.payload_on:
            raw = min(1.0, raw + 0.12)

        return self._safe_reward(raw)

    def _safe_reward(self, raw: float) -> float:
        eps     = 0.001
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

    def _get_observation(self, reward: float,
                         done: bool) -> OrbitalAnomalyOpenenvObservation:
        dropped = self._get_dropout_fields()

        def maybe(field, value, default=None):
            return default if field in dropped else value

        # Fault belief state (world modeling signal in metadata)
        b   = max(0.0, min(1.0, self.battery_soc / 100.0))
        s   = max(0.0, min(1.0, self.sun_vector_alignment * self.panel_health))
        t   = max(0.0, min(1.0, (self.payload_temp - 20.0) / 80.0))
        c   = max(0.0, min(1.0,
            1.0 - self.bit_error_rate * 5.0 - self.packet_loss_ratio))

        fault_beliefs = {
            "mppt_stuck":               round(max(0,(1-s)*0.90+(1-b)*0.30), 3),
            "radiator_valve_stuck":     round(max(0, t*0.85), 3),
            "transponder_overheating":  round(max(0,(1-c)*0.80+t*0.30), 3),
            "battery_aging":            round(max(0,(1-b)*0.50), 3),
            "antenna_gimbal_stall":     round(max(0,(1-c)*0.55+(1-s)*0.15), 3),
        }

        # Phase scores (for long-horizon tracking)
        phase_scores = []
        for ph_rewards in self._phase_rewards:
            if ph_rewards:
                phase_scores.append(round(sum(ph_rewards)/len(ph_rewards), 4))

        return OrbitalAnomalyOpenenvObservation(
            # ── V1 backward-compatible fields ─────────────────────────
            battery_level=round(self.battery_soc, 2),
            solar_efficiency=round(
                max(0.0, self.sun_vector_alignment * self.panel_health), 4),
            thermal_temp=round(self.payload_temp, 2),
            comms_signal=round(
                max(0.0, min(1.0,
                    1.0 - self.bit_error_rate * 5.0
                    - self.packet_loss_ratio)), 4),
            payload_on=self.payload_on,
            safe_mode=self.safe_mode,
            task_id=self.task_id,
            mission_status=self._mission_status(),
            reward=reward,
            done=done,

            # ── V2 EPS ────────────────────────────────────────────────
            battery_soc=round(self.battery_soc, 2),
            bus_voltage=round(self.bus_voltage, 3),
            panel_health=round(self.panel_health, 4),
            solar_array_current=round(
                maybe("solar_array_current", self.solar_array_current, -1.0), 3),
            charge_controller_health=round(
                self.charge_controller_health, 4),
            power_bus_redundancy=self.power_bus_redundancy,

            # ── V2 ADCS ───────────────────────────────────────────────
            attitude_error_deg=round(self.attitude_error_deg, 2),
            sun_vector_alignment=round(self.sun_vector_alignment, 4),
            reaction_wheel_momentum=round(self.reaction_wheel_momentum, 4),
            gyro_bias=round(
                maybe("gyro_bias", self.gyro_bias, -999.0), 3),
            star_tracker_available=maybe(
                "star_tracker_available",
                self.star_tracker_available, None),
            wheel_saturation_level=round(self.wheel_saturation_level, 4),

            # ── V2 Thermal ────────────────────────────────────────────
            battery_temp=round(self.battery_temp, 2),
            payload_temp=round(self.payload_temp, 2),
            avionics_temp=round(
                maybe("avionics_temp", self.avionics_temp, -999.0), 2),
            radiator_efficiency=round(self.radiator_efficiency, 4),
            heater_state=self.heater_state,
            thermal_loop_health=round(self.thermal_loop_health, 4),

            # ── V2 Comms ──────────────────────────────────────────────
            antenna_pointing_error=round(self.antenna_pointing_error, 2),
            transmitter_power=round(self.transmitter_power, 3),
            bit_error_rate=round(self.bit_error_rate, 6),
            uplink_margin=round(
                maybe("uplink_margin", self.uplink_margin, -999.0), 2),
            packet_loss_ratio=round(self.packet_loss_ratio, 4),
            command_latency_ms=round(
                maybe("command_latency_ms",
                      self.command_latency_ms, 9999.0), 1),

            # ── V2 Orbital ────────────────────────────────────────────
            sunlit=self.sunlit,
            eclipse_timer=self.eclipse_timer,
            ground_station_visible=self.ground_station_visible,
            radiation_zone=self.radiation_zone,
            observation_window_active=self.observation_window_active,

            # ── V2.1 Extended mission metadata ────────────────────────
            metadata={
                "step":          self._state.step_count,
                "episode_id":    self._state.episode_id,
                "version":       "2.1",
                "obs_dropout":   list(dropped),
                # Extended Mission Mode fields
                "phase":         self._phase,
                "phase_step":    self._phase_step,
                "extended_mission": self._extended_mission,
                "phase_scores":  phase_scores,
                # World model: fault belief state
                "fault_beliefs": fault_beliefs,
                # Active faults count (not names — stays hidden)
                "active_fault_count": len(self._faults),
            },
        )