# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
Utility functions for creating schedules and samplers from config.
"""

from typing import Any

import torch
from omegaconf import DictConfig

from .samplers.base import Sampler
from .samplers.euler import EulerSampler
from .schedules.base import Schedule
from .schedules.lerp import LinearInterpolationSchedule
from .timesteps.base import SamplingTimesteps
from .timesteps.sampling.trailing import UniformTrailingSamplingTimesteps


def create_schedule_from_config(
    config: DictConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Schedule:
    """
    Create a schedule from configuration.
    """
    if config.type == "lerp":
        return LinearInterpolationSchedule(T=config.get("T", 1.0))

    raise NotImplementedError


def create_sampler_from_config(config: DictConfig, **kwargs: Any) -> Sampler:
    """
    Create a sampler from configuration.
    """
    schedule = kwargs.pop("schedule", None)
    timesteps = kwargs.pop("timesteps", None)
    device = kwargs.pop("device", torch.device("cpu"))

    if config.type == "euler":
        sampler = EulerSampler(
            prediction_type=config.prediction_type,
            schedule_t=schedule.T if schedule is not None else 1000.0,
            timesteps_steps=len(timesteps) if timesteps is not None else 1000,
            device=device,
            **kwargs,
        )
        if schedule is not None:
            sampler.schedule = schedule
        if timesteps is not None:
            sampler.timesteps = timesteps
        return sampler
    raise NotImplementedError


def create_sampling_timesteps_from_config(
    config: DictConfig,
    schedule: Schedule,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SamplingTimesteps:
    if config.type == "uniform_trailing":
        return UniformTrailingSamplingTimesteps(
            T=schedule.T,
            steps=config.steps,
            shift=config.get("shift", 1.0),
            device=device,
        )
    raise NotImplementedError
