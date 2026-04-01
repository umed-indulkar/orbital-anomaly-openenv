# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Orbital Anomaly Openenv Environment.

The orbital_anomaly_openenv environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class OrbitalAnomalyOpenenvAction(Action):
    """Action for the Orbital Anomaly Openenv environment - just a message to echo."""

    message: str = Field(..., description="Message to echo back")


class OrbitalAnomalyOpenenvObservation(Observation):
    """Observation from the Orbital Anomaly Openenv environment - the echoed message."""

    echoed_message: str = Field(default="", description="The echoed message")
    message_length: int = Field(default=0, description="Length of the echoed message")
