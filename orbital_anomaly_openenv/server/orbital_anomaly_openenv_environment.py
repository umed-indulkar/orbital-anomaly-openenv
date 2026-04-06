# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Orbital Satellite Anomaly Response Environment.

A realistic spacecraft operations simulator where the agent must diagnose
telemetry anomalies and apply mission control actions to stabilize the satellite.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        OrbitalAnomalyOpenenvAction,
        OrbitalAnomalyOpenenvObservation,
    )
except ImportError:
    from models import (
        OrbitalAnomalyOpenenvAction,
        OrbitalAnomalyOpenenvObservation,
    )


class OrbitalAnomalyOpenenvEnvironment(Environment):
    """
    Satellite anomaly response simulator.

    The agent acts as mission control and must stabilize battery,
    thermal, and communications telemetry.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_id = "easy"

        self.battery_level = 100.0
        self.solar_efficiency = 1.0
        self.thermal_temp = 40.0
        self.comms_signal = 1.0
        self.payload_on = True
        self.safe_mode = False

    def reset(self) -> OrbitalAnomalyOpenenvObservation:
        """
        Reset into a new anomaly scenario.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.task_id = random.choice(["easy", "medium", "hard"])

        if self.task_id == "easy":
            self.battery_level = 42.0
            self.solar_efficiency = 0.25
            self.thermal_temp = 48.0
            self.comms_signal = 0.95
            self.payload_on = True
            self.safe_mode = False

        elif self.task_id == "medium":
            self.battery_level = 58.0
            self.solar_efficiency = 0.45
            self.thermal_temp = 82.0
            self.comms_signal = 0.80
            self.payload_on = True
            self.safe_mode = False

        else:  # hard
            self.battery_level = 35.0
            self.solar_efficiency = 0.15
            self.thermal_temp = 90.0
            self.comms_signal = 0.55
            self.payload_on = True
            self.safe_mode = False

        return self._get_observation(reward=0.0, done=False)

    def step(
        self, action: OrbitalAnomalyOpenenvAction
    ) -> OrbitalAnomalyOpenenvObservation:  # type: ignore[override]
        """
        Execute a mission control action.
        """
        self._state.step_count += 1

        self._apply_action(action.action_type)
        self._physics_update()

        reward = self._compute_reward()
        done = reward >= 0.92 or self._state.step_count >= 12

        return self._get_observation(reward=reward, done=done)

    def _apply_action(self, action_type: str):
        """
        Apply spacecraft corrective action.
        """
        if action_type == "rotate_to_sun":
            self.solar_efficiency = min(1.0, self.solar_efficiency + 0.35)

        elif action_type == "disable_payload":
            self.payload_on = False
            self.thermal_temp -= 10

        elif action_type == "reboot_comms":
            self.comms_signal = min(1.0, self.comms_signal + 0.25)

        elif action_type == "enter_safe_mode":
            self.safe_mode = True
            self.payload_on = False
            self.thermal_temp -= 6
            self.comms_signal = min(1.0, self.comms_signal + 0.1)

        elif action_type == "switch_power_bus":
            self.battery_level = min(100.0, self.battery_level + 8)

    def _physics_update(self):
        """
        Hidden subsystem evolution.
        """
        # Base spacecraft load
        self.battery_level += self.solar_efficiency * 5
        self.battery_level -= 3

        # Payload creates thermal + power load
        if self.payload_on:
            self.thermal_temp += 4
            self.battery_level -= 2
        else:
            self.thermal_temp -= 2

        # Thermal damage affects communications
        if self.thermal_temp > 85:
            self.comms_signal -= 0.08

        # Safe mode slowly improves stability
        if self.safe_mode:
            self.thermal_temp -= 1
            self.comms_signal += 0.03

        # Clamp ranges
        self.battery_level = max(0.0, min(100.0, self.battery_level))
        self.thermal_temp = max(0.0, min(120.0, self.thermal_temp))
        self.comms_signal = max(0.0, min(1.0, self.comms_signal))

    def _compute_reward(self) -> float:
        """
        Dense reward in 0–1 range.
        """
        battery_score = self.battery_level / 100
        thermal_score = max(0.0, 1 - self.thermal_temp / 100)
        comms_score = self.comms_signal

        reward = (battery_score + thermal_score + comms_score) / 3
        return round(max(0.0, min(1.0, reward)), 3)

    def _mission_status(self) -> str:
        """
        Human-readable mission health.
        """
        if self.battery_level < 25 or self.thermal_temp > 95:
            return "critical"
        if self.battery_level < 45 or self.thermal_temp > 80:
            return "warning"
        return "stable"

    def _get_observation(
        self, reward: float, done: bool
    ) -> OrbitalAnomalyOpenenvObservation:
        """
        Build typed telemetry observation.
        """
        return OrbitalAnomalyOpenenvObservation(
            battery_level=self.battery_level,
            solar_efficiency=self.solar_efficiency,
            thermal_temp=self.thermal_temp,
            comms_signal=self.comms_signal,
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

    @property
    def state(self) -> State:
        return self._state