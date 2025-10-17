import re
from typing import Dict, Tuple, Optional
import json
import re
import torch
from typing import Dict, Optional

import logging
# 过滤transformers库中关于loss_type的警告
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def remove_prefix(text:str):
    pattern = re.compile(
        r'<\|im_start\|>system.*?<\|im_start\|>user.*?<\|im_start\|>assistant',
        re.DOTALL
    )
    try:
        cleaned_text = pattern.sub('',text)
        if cleaned_text != text:
            return cleaned_text.strip()
        else:
            return text
    except re.error:
        return text

def get_single_eval(ground_truth: str, answer_text:str):
    ground_truth = ground_truth.lower().strip()
    answer_text = answer_text.lower().strip()

    if ground_truth == answer_text:
        answer_score = 1.0
    else:
        answer_score = -1.0

    return answer_score


def extract_solution(solution_str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    final_answer = matches[-1].group(1).strip()

    think_pattern = r'<think>(.*?)</think>'
    think_matches = list(re.finditer(think_pattern, solution_str, re.DOTALL))        
    final_think = think_matches[-1].group(1).strip()

    return final_think, final_answer

def extract_ground_truth(ground_truth):
    """Extracts the final answer from the model's response string.
    
    Args:
        ground_truth: Ground truth string from the dataset
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, ground_truth, re.DOTALL))
    final_answer = matches[-1].group(1).strip()

    think_pattern = r'<think>(.*?)</think>'
    think_matches = list(re.finditer(think_pattern, ground_truth, re.DOTALL))       
    final_think = think_matches[-1].group(1).strip()


    return final_think, final_answer

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    # print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        if count != expected_count:
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        validation_passed = False
    else:
        pass

    return validation_passed
def validate_response_structure_answer_first(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    # print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'answer_start': ('<answer>', 1),
        'answer_end': ('<//answer>', 1),
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        if count != expected_count:
            validation_passed = False

    # Verify tag order
    if (positions['answer_start'] > positions['answer_end'] or
        positions['answer_end'] > positions['think_start'] or
        positions['think_start'] > positions['think_end']):
        validation_passed = False
    else:
        pass

    return validation_passed

def compute_score(solution_str: str, 
                 ground_truth: str,
                 format_reward: int = 0.1,
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """

    # Extract model answer
    solution_str = remove_prefix(solution_str)
    think_text, answer_text = extract_solution(solution_str)

    # Validate response structure
    format_correct = validate_response_structure(solution_str)
    format_score = format_reward if format_correct else -abs(format_reward * 10)

    # Validate answer content
    answer_score = 0
    
    if format_correct and answer_text:
        
        answer_score = get_single_eval(ground_truth, answer_text)

            
    else:
        answer_score = -1.0

    total_score = format_score + answer_score

    print("\n" + "-"*60)
    print(" Processing New Seq ".center(60, '='))
    print(f"[Solution]")
    print(f"Predict: \n{solution_str}")

    print(f"\n[Format validation] {'PASS' if format_correct else 'FAIL'}")
    if format_correct and answer_text:
        print(f"\n[Content Validation]")
        print(f"Expected: \n{ground_truth}")
        print(f"[Answer] Predict: {answer_text}, Score: {answer_score}")
    else:
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    print("\n" + "-"*60)
    print(f" Final Score ".center(60, '-'))
    print(f"Format: {format_score}")
    print(f"Answer: {answer_score}")
    print(f"Total: {total_score}")
    print("="*60 + "\n")

    return total_score