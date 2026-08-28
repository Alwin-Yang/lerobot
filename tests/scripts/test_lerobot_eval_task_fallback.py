#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import pytest

from lerobot.scripts.lerobot_eval import _get_task_descriptions


class _FakeVectorEnv:
    def __init__(self, task_description, task, num_envs=2):
        self.num_envs = num_envs
        self._responses = {"task_description": task_description, "task": task}

    def call(self, name):
        response = self._responses[name]
        if isinstance(response, BaseException):
            raise response
        return response


def test_task_description_takes_precedence_over_task_and_fallback():
    env = _FakeVectorEnv(["description 0", "description 1"], ["task 0", "task 1"])

    assert _get_task_descriptions(env, "fallback") == ["description 0", "description 1"]


def test_task_attribute_is_used_when_description_is_unavailable():
    env = _FakeVectorEnv(AttributeError(), ["task 0", "task 1"])

    assert _get_task_descriptions(env, "fallback") == ["task 0", "task 1"]


@pytest.mark.parametrize("fallback_task", ["dataset task", None])
def test_dataset_task_fallback_is_broadcast_when_env_has_no_task_attributes(fallback_task):
    env = _FakeVectorEnv(AttributeError(), NotImplementedError(), num_envs=3)

    assert _get_task_descriptions(env, fallback_task) == [fallback_task or ""] * 3
