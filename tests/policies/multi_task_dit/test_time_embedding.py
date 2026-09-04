#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: E402

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
    DiffusionTransformer,
    NormalizedSinusoidalPosEmb,
    SinusoidalPosEmb,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def make_config(objective: str) -> MultiTaskDiTConfig:
    return MultiTaskDiTConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        objective=objective,
        timestep_embedding_type="openpi" if objective == "flow_matching" else "lerobot",
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        timestep_embed_dim=8,
    )


def test_objective_uses_matching_timestep_embedding():
    """FM uses an OpenPI-style [0, 1] embedding while DDPM keeps the LeRobot embedding."""
    fm_model = DiffusionTransformer(make_config("flow_matching"), conditioning_dim=8)
    assert isinstance(fm_model.time_mlp[0], NormalizedSinusoidalPosEmb)

    ddpm_model = DiffusionTransformer(make_config("diffusion"), conditioning_dim=8)
    assert isinstance(ddpm_model.time_mlp[0], SinusoidalPosEmb)


def test_old_flow_matching_checkpoint_keeps_lerobot_embedding():
    config = make_config("flow_matching")
    config.timestep_embedding_type = "lerobot"
    model = DiffusionTransformer(config, conditioning_dim=8)

    assert isinstance(model.time_mlp[0], SinusoidalPosEmb)


def test_normalized_timestep_embedding_matches_openpi_periods():
    """The continuous FM embedding uses 2*pi*t/period over the configured period range."""
    embedding = NormalizedSinusoidalPosEmb(dim=4, min_period=4e-3, max_period=4.0)
    timesteps = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    actual = embedding(timesteps)
    periods = torch.tensor([4e-3, 4.0], dtype=torch.float32)
    angles = timesteps[:, None] * (2 * torch.pi / periods[None, :])
    expected = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    assert actual.dtype == timesteps.dtype
    assert torch.allclose(actual, expected, atol=1e-4)


def test_flow_matching_time_embedding_supports_backward():
    model = DiffusionTransformer(make_config("flow_matching"), conditioning_dim=8)
    sample = torch.randn(2, 4, 2)
    timestep = torch.tensor([0.2, 0.8])
    conditioning = torch.randn(2, 8)

    output = model(sample, timestep, conditioning)
    output.square().mean().backward()

    assert output.shape == sample.shape
    assert model.time_mlp[1].weight.grad is not None
