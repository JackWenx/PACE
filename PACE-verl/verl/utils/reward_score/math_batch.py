# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
import math
import re
import ray
import time
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
def extract_answer(input_str):
    index = input_str.find('</think>')
    if index == -1:
        return "None"
    return input_str[index + len('</think>'):]


@ray.remote(num_cpus=1, memory=4
 * 1024 * 1024 * 1024)
def single_compute_score(solution_str, ground_truth, format_required):
    # print('singlecomputescore')
    gold_parsed = [ground_truth]


    solution_only = extract_answer(solution_str)
    # We require the answer to be provided in correct latex (no malformed operators)
    answer_parsed = parse(
        solution_only,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=False,
                    boxed="last",
                    units=True,
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    if len(answer_parsed) > 0:
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        reward_verl = float(is_equiv(answer_parsed[-1],gold_parsed[-1]))
        reward = reward_verl
    else:
        # If the gold solution is not parseable, we reward 1 to skip this example
        reward = 0.0
        # print("Failed to parse solution")

    if format_required:
        format_reward = format_reward_func(solution_str)
        tag_count_reward = tag_count_reward_func(solution_str)
        score = {
            "score": reward,
            "format_reward": format_reward,
            "answer_reward": reward,
            "comparative_reward": 0.0,
            "tag_count_reward": tag_count_reward
        }  
    else:
        score = reward  

    return score

def parallel_compute_score(predictions, references, format_required=False, timeout_all=300.0):
    scores = []
    total_tasks = len(predictions)
    start_time = time.time()

    assert ray.is_initialized(), "Ray must be initialized before calling parallel_compute_score"
    # print('start submitting75')

    format_required = True
    tasks_async = [
        single_compute_score.remote(solution_str, ground_truth, format_required)
        for solution_str, ground_truth in zip(predictions, references)
    ]
    # print('start submitting')

    remaining = tasks_async
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=min(len(remaining), 1000), timeout=3.0)
        completed = total_tasks - len(remaining)
        elapsed = time.time() - start_time
        eta = (elapsed / max(1, completed)) * (total_tasks - completed) if completed > 0 else 0

        print(f"MATH REWARD Progress: {completed}/{total_tasks} tasks completed ({(completed/total_tasks)*100:.1f}%) "
              f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed))} elapsed, "
              f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}]")

    try:
        scores = ray.get(tasks_async, timeout=timeout_all)  
    except ray.exceptions.RayTaskError as e:
        raise RuntimeError(f"Ray task error occurred: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in fetching Ray task results: {e}")

    return scores

def compute_score(predictions, references, format_required=True, problems=None):
    scores = []

    scores = parallel_compute_score(predictions, references, format_required)
    return scores

    # print(f'computescore {len(predictions)}')
    # for solution_str, ground_truth in zip(predictions, references):
    #     gold_parsed = [ground_truth]

    #     # We require the answer to be provided in correct latex (no malformed operators)
    #     answer_parsed = parse(
    #         solution_str,
    #         extraction_config=[
    #             LatexExtractionConfig(
    #                 normalization_config=NormalizationConfig(
    #                     nits=False,
    #                     malformed_operators=False,
    #                     basic_latex=True,
    #                     equations=False,
    #                     boxed="last",
    #                     units=True,
    #                 ),
    #                 # Ensures that boxed is tried first
    #                 boxed_match_priority=0,
    #                 try_extract_without_anchor=False,
    #             )
    #         ],
    #         extraction_mode="first_match",
    #     )
    #     if len(answer_parsed) > 0:
    #         # Reward 1 if the content is the same as the ground truth, 0 otherwise
    #         reward_verl = float(is_equiv(answer_parsed[-1],gold_parsed[-1]))
    #         reward = reward_verl
    #     else:
    #         # If the gold solution is not parseable, we reward 1 to skip this example
    #         reward = 0.0
    #         # print("Failed to parse solution")
        

    #     if format_required:
    #         format_reward = format_reward_func(solution_str)
    #         tag_count_reward = tag_count_reward_func(solution_str)
    #         score = {
    #             "score": reward + format_reward + tag_count_reward,
    #             "format_reward": format_reward,
    #             "answer_reward": reward,
    #             "comparative_reward": 0.0,
    #             "tag_count_reward": tag_count_reward
    #         }
    #     else:
    #         score = reward

    #     scores.append(score)

    # return scores

def format_reward_func(solution_str):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    # pattern = r"^<think>.*?</think>\s*\n*<answer>.*?</answer>$"
    if solution_str is None:
        return 0.0
    pattern = r"^<think>.*?</think>.*$"
    if solution_str.strip().endswith('<|im_end|>'):
        solution_str = solution_str.strip()[:-len('<|im_end|>')]
    if solution_str.strip().endswith('<|endoftext|>'):
        solution_str = solution_str.strip()[:-len('<|endoftext|>')]        
    match = re.match(pattern, solution_str.strip(), re.DOTALL | re.MULTILINE)
    if match:
        reward = 0.5
    else:
        reward = 0.0
    return reward

def tag_count_reward_func(solution_str):
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """
    if solution_str is None:
        return 0.0
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>") == 1:
            count += 0.25
        if text.count("</think>") == 1:
            count += 0.25
        # if text.count("<answer>") == 1:
        #     count += 0.25
        # if text.count("</answer>") == 1:
        #     count += 0.25
        return count

    return count_tags(solution_str)

# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
