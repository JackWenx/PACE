# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import ray
from collections import defaultdict


@ray.remote
def compute_wrapper(func,args):
    data_source, Values, format_required = args
    return func(
        data_source=data_source,
        solution_str=Values['predictions'],
        ground_truth=Values['references'],
        format_required=format_required,
        extra_info=Values['extra_info'],
    ), Values['idx']
class GenerativeRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source',
                 format_required=False,
                 max_resp_len=None,
                 overlong_buffer_cfg=None,
                 use_length_reward=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.format_required = format_required

        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"



    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # These lists are still useful for the final processing loop
        problems = []
        references = []
        predictions = []
        valid_response_length_list = []
        data_source_lst = []

        # This part remains the same: collect and group data first
        Values_by_datasource = defaultdict(lambda: defaultdict(list))
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            Values_by_datasource[data_source]['predictions'].append(response_str)
            Values_by_datasource[data_source]['references'].append(ground_truth)
            Values_by_datasource[data_source]['idx'].append(i)
            Values_by_datasource[data_source]['extra_info'].append(extra_info)

            # Still need these for printing and final reward assignment
            problems.append(prompt_str)
            predictions.append(response_str)
            references.append(ground_truth)
            valid_response_length_list.append(valid_response_length)
            data_source_lst.append(data_source)


        ## --- MODIFICATION START --- ##
        # Instead of preparing one task per data_source, prepare one task per item.
        tasks = []
        for ds, values in Values_by_datasource.items():
            # Get the number of items for this data source
            num_items = len(values['predictions'])
            
            # Create a separate task for each item
            for i in range(num_items):
                # Package the single item's data. Crucially, keep them as lists of one element
                # to maintain compatibility with `compute_score` and the result processing loop.
                single_item_values = {
                    'predictions': [values['predictions'][i]],
                    'references': [values['references'][i]],
                    'extra_info': [values['extra_info'][i]],
                    'idx': [values['idx'][i]], # The index must also be in a list
                }
                
                # The arguments for compute_wrapper
                task_args = (ds, single_item_values, self.format_required)
                tasks.append(task_args)

        ## --- MODIFICATION END --- ##

        scores = []
        reorder_idx = []

        # The rest of the code works without changes because the data structure
        # returned by the remote function is the same (a list of scores and a list of indices).
        futures = [compute_wrapper.remote(self.compute_score, task) for task in tasks]
        results = ray.get(futures)
        for part_scores, part_idx in results:
            scores.extend(part_scores)
            reorder_idx.extend(part_idx)

        # Reorder the scores to match the original data order
        scores = [s for _, s in sorted(zip(reorder_idx, scores))]

        # Final processing loop remains the same
        for i in range(len(data)):
            valid_response_length = valid_response_length_list[i]
            score = scores[i]
            data_source = data_source_lst[i]

            if isinstance(score, dict):
                reward = score["score"]
                reward_extra_info["answer_reward"].append(score["answer_reward"])
                for key, value in score.items():
                    for ds in Values_by_datasource.keys():
                        reward_extra_info[f"{ds}_{key}"].append(-100)                    
                    reward_extra_info[f"{data_source}_{key}"][-1] = value
            else:
                reward = score

            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                
                if self.overlong_buffer_cfg.log:
                    for ds in Values_by_datasource.keys():
                        reward_extra_info[f"{ds}_overlong_reward"].append(-100)     
                        reward_extra_info[f"{ds}_overlong"].append(-100)                    
                    reward_extra_info[f"{data_source}_overlong_reward"][-1] = overlong_reward
                    reward_extra_info[f"{data_source}_overlong"][-1] = overlong_reward < 0

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[data_source]", data_source)
                print("[prompt]", problems[i])
                print("[response]", predictions[i]) 
                print("[ground_truth]", references[i])
                print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor