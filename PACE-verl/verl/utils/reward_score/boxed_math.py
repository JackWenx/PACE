from .math_verify import compute_score
from typing import Dict, Tuple, Optional
from collections import defaultdict
import re

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""
def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True
    
    # Check required tags
    tags = {
        'think_start': ('<think>', 2),
        'think_end': ('</think>', 2),
        # 'answer_start': ('<answer>', 1),
        # 'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    # 先不看answer标签
    if (positions['think_start'] > positions['think_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed


def compute_score_boxed(model_output: str, ground_truth: str):

    format_reward = -0.5

    if validate_response_structure(model_output):
        format_reward = 0.5

    answer_reward = compute_score(model_output, ground_truth)
    if answer_reward != 1.0:
        answer_reward = -1.0
    return format_reward + answer_reward