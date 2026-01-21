# Copyright 2024 PRIME team and/or its affiliates
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

from .registry import get_reward_manager_cls, register  # noqa: I001
from .batch import BatchRewardManager
from .dapo import DAPORewardManager
from .naive import NaiveRewardManager
from .prime import PrimeRewardManager
from .ShortRL import ShortRLRewardManager
from .l1 import L1RewardManager
from .generative_length import GenerativeLengthRewardManager
from .generative import GenerativeRewardManager
from .generative_length_length_penalty import GenerativeLengthRewardManagerAblation
from .generative_length_without_control import GenerativeLengthRewardManagerNoControl

# Note(haibin.lin): no need to include all reward managers here in case of complicated dependencies
__all__ = [
    "BatchRewardManager",
    "DAPORewardManager",
    "NaiveRewardManager",
    "PrimeRewardManager",
    "ShortRLRewardManager",
    "L1RewardManager",
    "GenerativeLengthRewardManager",
    'GenerativeRewardManager',
    "register",
    "get_reward_manager_cls",
    "GenerativeLengthRewardManagerAblation",
    "GenerativeLengthRewardManagerNoControl",
]
