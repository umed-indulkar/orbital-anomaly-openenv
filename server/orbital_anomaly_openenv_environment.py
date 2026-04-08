# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Orbital Satellite Anomaly Response Environment.

A realistic spacecraft operations simulator where the agent must diagnose
telemetry anomalies and apply mission-control actions to stabilize the satellite.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    OrbitalAnomalyOpenenvAction,
    OrbitalAnomalyOpenenvObservation,
)

TASK_IDS = ["easy", "medium", "hard"]


class OrbitalAnomalyOpenenvEnvironment(Environment):
    """
    Satellite anomaly response simulator.

    The agent acts as mission control and must stabilize battery,
    thermal, communication, and payload systems across three
    deterministic benchmark tasks: easy, medium, and hard.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Class-level counter so fresh instances still cycle tasks
    _global_reset_count: int = 0

    def __init__(self):
        self._state = State(
            episode_id=str(uuid4()),
            step_count=0,
        )

        self.task_id: str = "easy"
        self.battery_level: float = 100.0
        self.solar_efficiency: float = 1.0
        self.thermal_temp: float = 40.0
        self.comms_signal: float = 1.0
        self.payload_on: bool = True
        self.safe_mode: bool = False

    # ------------------------------------------------------------------
    # OpenEnv Interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str | None = None) -> OrbitalAnomalyOpenenvObservation:
        """
        Reset into a deterministic benchmark task.

        If ``task_id`` is explicitly provided (easy | medium | hard) the
        environment is loaded directly into that scenario — this satisfies
        the Phase-2 per-task grader which calls reset(task_id=...).

        If ``task_id`` is None the counter cycles easy → medium → hard so
        that sequential grader calls without an explicit id still cover all
        three tasks.
        """
        self._state = State(
            episode_id=str(uuid4()),
            step_count=0,
        )

        if task_id in TASK_IDS:
            self.task_id = task_id
        else:
            # Cycle deterministically across tasks
            self.task_id = TASK_IDS[
                self.__class__._global_reset_count % 3
            ]

        self.__class__._global_reset_count += 1

        self._load_task(self.task_id)

        return self._get_observation(reward=self._safe_reward(0.35), done=False)

    def step(
        self,
        action: OrbitalAnomalyOpenenvAction,
    ) -> OrbitalAnomalyOpenenvObservation:
        """Execute one mission-control action and advance the simulation."""
        self._state.step_count += 1

        self._apply_action(action.action_type)
        self._physics_update()

        reward = self._compute_reward()

        done = (
            (
                self.battery_level >= 70
                and self.thermal_temp <= 60
                and self.comms_signal >= 0.90
            )
            or self._state.step_count >= 12
        )

        return self._get_observation(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Task Initialisation
    # ------------------------------------------------------------------

    def _load_task(self, task_id: str) -> None:
        """Load deterministic initial conditions for each task scenario."""
        if task_id == "easy":
            # Primary anomaly: solar misalignment + low battery
            self.battery_level = 42.0
            self.solar_efficiency = 0.25
            self.thermal_temp = 48.0
            self.comms_signal = 0.95
            self.payload_on = True
            self.safe_mode = False

        elif task_id == "medium":
            # Primary anomaly: thermal overload with moderate battery
            self.battery_level = 58.0
            self.solar_efficiency = 0.45
            self.thermal_temp = 82.0
            self.comms_signal = 0.80
            self.payload_on = True
            self.safe_mode = False

        else:  # hard
            # Cascading failure: low battery + solar + thermal + comms
            self.battery_level = 30.0
            self.solar_efficiency = 0.20
            self.thermal_temp = 92.0
            self.comms_signal = 0.55
            self.payload_on = True
            self.safe_mode = False

    # ------------------------------------------------------------------
    # Action Application
    # ------------------------------------------------------------------

    def _apply_action(self, action_type: str) -> None:
        """Apply corrective spacecraft recovery action."""
        if action_type == "rotate_to_sun":
            self.solar_efficiency = min(1.0, self.solar_efficiency + 0.35)

        elif action_type == "disable_payload":
            self.payload_on = False
            self.thermal_temp = max(0.0, self.thermal_temp - 10)

        elif action_type == "reboot_comms":
            self.comms_signal = min(1.0, self.comms_signal + 0.25)

        elif action_type == "enter_safe_mode":
            self.safe_mode = True
            self.payload_on = False
            self.thermal_temp = max(0.0, self.thermal_temp - 6)
            self.comms_signal = min(1.0, self.comms_signal + 0.10)

        elif action_type == "switch_power_bus":
            self.battery_level = min(100.0, self.battery_level + 8)

        # "noop" → no change

    # ------------------------------------------------------------------
    # Physics Simulation
    # ------------------------------------------------------------------

    def _physics_update(self) -> None:
        """Hidden spacecraft subsystem evolution (causal dynamics)."""
        # Solar charging minus baseline drain
        self.battery_level += self.solar_efficiency * 5
        self.battery_level -= 3

        # Payload draws power and generates heat
        if self.payload_on:
            self.thermal_temp += 4
            self.battery_level -= 2
        else:
            self.thermal_temp -= 2

        # Thermal stress degrades comms hardware
        if self.thermal_temp > 85:
            self.comms_signal -= 0.08

        # Safe mode gradually stabilises thermal & comms
        if self.safe_mode:
            self.thermal_temp -= 1
            self.comms_signal = min(1.0, self.comms_signal + 0.03)

        # Clamp all values to operational ranges
        self.battery_level = max(0.0, min(100.0, self.battery_level))
        self.thermal_temp = max(0.0, min(120.0, self.thermal_temp))
        self.comms_signal = max(0.0, min(1.0, self.comms_signal))

    # ------------------------------------------------------------------
    # Reward Computation
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        """
        Dense per-step reward strictly in the open interval (0, 1).

        Combines normalised battery health, inverse thermal load, and
        communication quality into a single survivability score.
        """
        battery_score = self.battery_level / 100.0
        thermal_score = max(0.0, 1.0 - self.thermal_temp / 100.0)
        comms_score = self.comms_signal

        raw = (battery_score + thermal_score + comms_score) / 3.0
        return self._safe_reward(raw)

    @staticmethod
    def _safe_reward(raw: float) -> float:
        """
        Map any float to the strict open interval (0.001, 0.999).

        This satisfies the Phase-2 grader requirement that scores must
        be strictly between 0 and 1 — never exactly 0.0 or 1.0.
        """
        eps = 0.001
        clamped = max(0.0, min(1.0, raw))
        return round(eps + clamped * (1.0 - 2 * eps), 4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mission_status(self) -> str:
        if self.battery_level < 25 or self.thermal_temp > 95:
            return "critical"
        if self.battery_level < 45 or self.thermal_temp > 80:
            return "warning"
        return "stable"

    def _get_observation(
        self,
        reward: float,
        done: bool,
    ) -> OrbitalAnomalyOpenenvObservation:
        return OrbitalAnomalyOpenenvObservation(
            battery_level=round(self.battery_level, 3),
            solar_efficiency=round(self.solar_efficiency, 4),
            thermal_temp=round(self.thermal_temp, 3),
            comms_signal=round(self.comms_signal, 4),
            payload_on=self.payload_on,
            safe_mode=self.safe_mode,
            task_id=self.task_id,
            mission_status=self._mission_status(),
            reward=reward,
            done=done,
            metadata={
                "step": self._state.step_count,
                "episode_id": self._state.episode_id,
            },
        )