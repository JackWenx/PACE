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

from cmath import pi
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import ray
from collections import defaultdict
import math
import numpy as np

from verl.workers.reward_manager import register


@ray.remote
def compute_wrapper(func,args):
    data_source, Values, format_required = args
    return func(
        data_source=data_source,
        solution_str=Values['predictions'],
        ground_truth=Values['references'],
        format_required=format_required,
        extra_info=Values['extra_info'],
        problem=Values['problems']
    ), Values['idx']


@register('generative_length')
class GenerativeLengthRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source',
                 format_required=False,
                 max_resp_len=None,
                 overlong_buffer_cfg=None,
                 use_length_reward=True) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.format_required = format_required

        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
        
        self.use_length_reward = use_length_reward


    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}


        problems = []
        references = []
        predictions = []
        valid_response_length_list = []
        data_source_lst = []
        Values_by_datasource = defaultdict()
        step_list = []
        for i in range(len(data)):
            data_item = data[i] 

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            # prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']


            if data_source not in Values_by_datasource.keys():
                Values_by_datasource[data_source] = defaultdict(list)

            Values_by_datasource[data_source]['problems'].append(data_item.non_tensor_batch['reward_model']['problem'])
            Values_by_datasource[data_source]['predictions'].append(response_str)
            Values_by_datasource[data_source]['references'].append(ground_truth)
            Values_by_datasource[data_source]['idx'].append(i)


            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            Values_by_datasource[data_source]['extra_info'].append(extra_info)

            problems.append(prompt_str)
            predictions.append(response_str)
            references.append(ground_truth)
            valid_response_length_list.append(valid_response_length)
            data_source_lst.append(data_source)
            step_list.append(data_item.meta_info.get('global_steps', 0))


        # Prepare tasks
        tasks = [
            (ds, Values_by_datasource[ds], self.format_required)
            for ds in Values_by_datasource.keys()
        ]

        scores = []
        reorder_idx = []


        futures = [compute_wrapper.remote(self.compute_score,task) for task in tasks]
        results = ray.get(futures)
        for part_scores, part_idx in results:
            scores.extend(part_scores)
            reorder_idx.extend(part_idx)


        
        scores = [s for _, s in sorted(zip(reorder_idx, scores))]
        
        
        problem_dict = defaultdict(list)
        answer_rewards = [score["answer_reward"] if isinstance(score, dict) else score for score in scores]
        for length, answer_reward, problem in zip(valid_response_length_list, answer_rewards, problems):
            problem_dict[problem].append((length, answer_reward))

        problem_stats = {}

        for problem, vals in problem_dict.items():
            lengths = [v[0] for v in vals]
            rewards = [v[1] for v in vals]

            avg_length = sum(lengths) / len(lengths)
            max_length = max(lengths)
            min_length = min(lengths)
            avg_acc = sum(rewards) / len(rewards)

            problem_stats[problem] = {
                'avg_length': avg_length,
                'max_length': max_length,
                'min_length': min_length,
                'avg_acc': avg_acc,
                'std_length': np.std(lengths)
            }


        def length_reward_func(length, answer_reward, max_length, min_length, avg_acc, step=0, max_step=600):
            """
            - length_reward = 0.5 - (length - min_length) / (max_length - min_length)
            """
            scale_factor = 1 - math.cos(avg_acc * (pi / 2))
            if max_length == min_length:
                return 0.0
            step_factor = max(0.0, min(1.0, (max_step - step) / max_step))
            print(f'step factor: {step_factor}')
            print(f'cur_step: {step}')
            length_reward = 0.5 - (length - min_length) / (max_length - min_length)
            new_length_reward = step_factor * scale_factor * length_reward
            if answer_reward == 0.0:
                return min(0, new_length_reward)
            return new_length_reward


        length_rewards = []
        for length, answer_reward, problem, step in zip(valid_response_length_list, answer_rewards, problems, step_list):
            stats = problem_stats[problem]
            lr = length_reward_func(length, answer_reward, stats['max_length'], stats['min_length'], stats['avg_acc'], step=step)
            length_rewards.append(lr)
        
        # print("Length rewards for each rollout:", length_rewards)            

        for i in range(len(data)):
            valid_response_length = valid_response_length_list[i]
            score = scores[i]
            data_source = data_source_lst[i]


            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                reward_extra_info["answer_reward"].append(score["answer_reward"])



                for key, value in score.items():
                    for ds in Values_by_datasource.keys():
                        reward_extra_info[f"{ds}_{key}"].append(-100)                    
                    reward_extra_info[f"{data_source}_{key}"][-1] = value
                    # reward_extra_info[key].append(value)
                
                # for key, value in reward_extra_info.items():
                #     print(f'{key} {len(value)}')
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

            if self.use_length_reward:
                reward += length_rewards[i]

                for ds in Values_by_datasource.keys():
                    reward_extra_info[f"{ds}_length_reward"].append(-100)     
                reward_extra_info[f"{data_source}_length_reward"][-1] = length_rewards[i]
                # reward_extra_info["length_reward"].append(length_rewards[i])

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
                print("[length reward]", length_rewards[i])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

        return reward_tensor
