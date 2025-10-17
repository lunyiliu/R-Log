import json
import re
import argparse
import sys
from pathlib import Path


# --- Utility Functions ---

def extract_solution(solution_str):
    """
    Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model.

    Returns:
        Extracted answer string.
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if not matches:
        # If no <answer> tags are found, return the original string
        return solution_str.strip()

    final_answer = matches[-1].group(1).strip()
    return final_answer


def validate_first_line(text):
    """
    Checks the first line of text for 'normal' or 'abnormal' keywords.
    Used for anomaly detection tasks.
    """
    first_line = text.split('\n')[0].lower()

    tokens = first_line.split()
    has_normal = 'normal' in tokens
    has_abnormal = 'abnormal' in tokens

    if has_normal and has_abnormal:
        return 'unknown'
    elif has_normal:
        return 'normal'
    elif has_abnormal:
        return 'abnormal'
    else:
        return 'unknown'


# --- Main Execution ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate a language model's output against a test set."
    )

    parser.add_argument(
        '--model', '-m',
        type=str, required=True,
        help="Model name (must exist in models_config.json)."
    )

    parser.add_argument(
        '--testset', '-t',
        type=str, required=True,
        help="Test set name (must exist in testsets_config.json)."
    )

    args = parser.parse_args()

    # --- Load configuration files ---
    config_dir = Path(__file__).parent

    try:
        with open(config_dir / 'models_config.json', 'r', encoding='utf-8') as f:
            model_paths = json.load(f)
        with open(config_dir / 'testsets_config.json', 'r', encoding='utf-8') as f:
            testsets_data = json.load(f)
    except FileNotFoundError as e:
        print(f"[Error] Missing configuration file: {e.filename}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[Error] Invalid JSON format in configuration files: {e}")
        sys.exit(1)

    # --- Validate arguments ---
    if args.model not in model_paths:
        print(f"[Error] Model '{args.model}' not found in models_config.json.")
        sys.exit(1)
    if args.testset not in testsets_data:
        print(f"[Error] Test set '{args.testset}' not found in testsets_config.json.")
        sys.exit(1)

    model_name = args.model
    testset_name = args.testset

    testset_info = testsets_data[testset_name]
    test_data_file = testset_info['path']
    testset_type = testset_info.get('type', '').lower()  # e.g., 'anomaly', 'parsing', etc.
    model_path = model_paths[model_name]

    # --- Output directory setup ---
    BASE_OUTPUT_DIR = Path('../LLaMA-Factory')
    infer_result_file = BASE_OUTPUT_DIR / f'original_output/{testset_name}_by_{model_name}.json'
    labelled_infer_result_file = BASE_OUTPUT_DIR / f'labelled_output/{testset_name}_by_{model_name}.json'
    diff_infer_result_file = BASE_OUTPUT_DIR / f'diff_output/{testset_name}_by_{model_name}.json'

    # --- Load test and inference data ---
    try:
        with open(infer_result_file, 'r', encoding='utf-8') as f:
            infer_result = json.load(f)
        with open(test_data_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError as e:
        print(f"[Error] Data file not found: {e.filename}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[Error] Failed to decode JSON: {e}")
        sys.exit(1)

    # --- Build label dictionary ---
    label_dict = {unit['input']: unit['output'] for unit in test_data if 'input' in unit and 'output' in unit}

    bio_checker_num = 0
    none_bio_num = 0

    # --- Process inference results ---
    for infer_unit in infer_result:
        key = infer_unit.get('input')
        if not key:
            continue

        infer_unit['label'] = label_dict.get(key, 'unknown')
        raw_output = infer_unit.get('output', '').strip()

        # Extract final answer content if <answer> tags exist
        final_answer = extract_solution(raw_output)
        infer_unit['output'] = final_answer

        # Apply binary validation for anomaly detection
        if testset_type == 'anomaly':
            infer_output_lower = final_answer.lower()
            if infer_output_lower not in ['normal', 'abnormal']:
                bio_checker_num += 1
                validated_output = validate_first_line(infer_output_lower)
                infer_unit['output'] = validated_output
                if validated_output == 'unknown':
                    none_bio_num += 1

                print(
                    f"\n[Binary normalization #{bio_checker_num}]"
                    f"\nInput key: {key}"
                    f"\nRaw output: {raw_output}"
                    f"\nNormalized output: {infer_unit['output']}"
                )

    print(f"\n[Summary] Number of 'unknown' outputs: {none_bio_num}")

    # --- Compute differences ---
    diff_result = [
        unit for unit in infer_result
        if unit.get('label') != unit.get('output')
    ]

    # --- Save processed results ---
    labelled_infer_result_file.parent.mkdir(parents=True, exist_ok=True)
    diff_infer_result_file.parent.mkdir(parents=True, exist_ok=True)

    with open(labelled_infer_result_file, 'w', encoding='utf-8') as f:
        json.dump(infer_result, f, ensure_ascii=False, indent=4)

    with open(diff_infer_result_file, 'w', encoding='utf-8') as f:
        json.dump(diff_result, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Processing complete.")
    print(f"   Labelled output → {labelled_infer_result_file}")
    print(f"   Diff output → {diff_infer_result_file}")
