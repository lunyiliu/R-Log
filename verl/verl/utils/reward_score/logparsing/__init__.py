import re
from typing import List, Tuple, Optional
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

# --- Subword and Filtering Utilities ---

def merge_subwords(subwords: List[str]) -> List[str]:
    """
    Merge BERT/WordPiece subwords into complete tokens by concatenating
    any subwords starting with '##' to their preceding tokens.
    """
    merged_words = []
    for subword in subwords:
        if subword.startswith("##"):
            if merged_words:
                merged_words[-1] += subword.replace("##", "")
            else:
                # Should not occur in normal cases, but included as a safeguard
                merged_words.append(subword.replace("##", ""))
        else:
            merged_words.append(subword)
    return merged_words


def filter_special_chars_for_F1(s: List[str]) -> List[str]:
    """
    Filter out special characters from a list of tokens,
    keeping only letters, digits, underscores, whitespace, and wildcards (*).
    """
    special_chars = r'[^\w\s\*]'
    filtered_str = [re.sub(special_chars, '', ele) for ele in s]
    filtered_str = [ele for ele in filtered_str if ele != '']
    return filtered_str


# --- Variable Extraction ---

def extract_variables(template: str, raw_log: str) -> List[str]:
    """
    Extract variable segments from the raw log based on the template,
    where variable positions are marked as <*> in the template.
    """
    # Replace '*' in the raw log with '#' to avoid confusion with <*> in templates.
    # Also escape regex-sensitive characters like () and [].
    raw_log = raw_log.replace('*', '#')
    template = template.replace('*', '#')
    template = template.replace('(', '\(').replace(')', '\)')
    template = template.replace('[', '\[').replace(']', '\]')

    # Compress consecutive <#> tokens into a single one to ensure regex stability
    template = re.sub(r"(<#>\s?)+", "<#>", template)

    # Replace <#> with a non-greedy regex group to extract variable values
    pattern = template.replace('<#>', '(.*?)')

    try:
        matches = re.findall(pattern, raw_log)
    except re.error as e:
        print(f"[Regex Error] Pattern: {pattern}, Error: {e}")
        return []

    if matches:
        # re.findall may return a string (single match) or a tuple (multiple matches)
        if isinstance(matches[0], str):
            return [matches[0].strip()]
        else:
            return [match.strip() for match in matches[0]]
    else:
        return []


# --- Core F1 Reward Calculation ---

def f1_reward(true_label: List[str], predicted_label: List[str], raw_logs: List[str]) -> float:
    """
    Compute a token-level F1-based reward.
    Each token is classified as either 'template' (fixed part) or 'variable' (changing part).
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    all_true_tokens = []
    all_predicted_tokens = []
    contains_variables = False

    for true, pred, log in zip(true_label, predicted_label, raw_logs):
        # 1. Preprocess model output by removing <think> and <answer> sections
        pred_content = re.sub(r'^<think>.*</think>\s*<answer>|<\/answer><\|endoftext\|>$', '', pred, flags=re.DOTALL).strip()

        # 2. Normalize true template
        true_norm = true.replace('<*>', '<#>').replace('*', '#')
        true_norm = re.sub(r"(<#>\s?)+", "<#>", true_norm)
        true_norm = true_norm.replace('<#>', '<*>')

        # 3. Normalize predicted template
        pred_norm = pred_content.replace('<*>', '<#>').replace('*', '#')
        pred_norm = re.sub(r"(<#>\s?)+", "<#>", pred_norm)
        pred_norm = pred_norm.replace('<#>', '<*>')

        # 4. Tokenize, merge subwords, and filter special characters
        true_token_list = filter_special_chars_for_F1(merge_subwords(tokenizer.tokenize(true_norm)))
        pred_token_list = filter_special_chars_for_F1(merge_subwords(tokenizer.tokenize(pred_norm)))
        pred_token_list_lower = [t.lower() for t in pred_token_list]

        # 5. Check whether the true template contains variables
        has_variables_in_true = '<*>' in true_norm
        if has_variables_in_true:
            contains_variables = True

        # 6. Classify true tokens
        true_token_classes = ["template" if t != '<*>' else "variable" for t in true_token_list]

        # 7. Classify predicted tokens based on matches to the true template
        predicted_token_classes = []
        variables = extract_variables(true, log)
        variable_index = 0

        for token in true_token_list:
            if token == '<*>':
                # For variable tokens: check if true variable values appear in the prediction
                if variable_index < len(variables) and variables[variable_index].strip().lower() in pred_content.lower() and '<*>' not in pred_norm:
                    predicted_token_classes.append("template")  # Correctly predicted variable
                else:
                    predicted_token_classes.append("variable")  # Variable mismatched or still predicted as <*>
                variable_index += 1
            else:
                # For fixed tokens: check if token appears in the prediction
                if token.lower() in pred_token_list_lower:
                    predicted_token_classes.append("template")
                else:
                    predicted_token_classes.append("variable")

        # 8. Handle inconsistent token lengths
        if len(true_token_classes) != len(predicted_token_classes):
            # Indicates structural mismatch in prediction
            return 0.0

        all_true_tokens.extend(true_token_classes)
        all_predicted_tokens.extend(predicted_token_classes)

    # 9. Choose positive label based on whether template contains variables
    pos_label = "variable" if contains_variables else "template"

    # 10. Handle edge cases where no variable tokens exist
    if pos_label == "variable" and not any(t == "variable" for t in all_true_tokens):
        return 1.0 if not any(t == "variable" for t in all_predicted_tokens) else 0.0

    if not all_true_tokens:
        return 1.0

    f1 = f1_score(all_true_tokens, all_predicted_tokens, pos_label=pos_label, zero_division=0)
    return f1


# --- Response Structure Extraction & Validation ---

def extract_solution(solution_str: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the content inside <think>...</think> and <answer>...</answer> blocks."""
    think_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'

    think_match = re.search(think_pattern, solution_str, re.DOTALL)
    answer_match = re.search(answer_pattern, solution_str, re.DOTALL)

    final_think = think_match.group(1).strip() if think_match else None
    final_answer = answer_match.group(1).strip() if answer_match else None

    return final_think, final_answer


def validate_response_structure(processed_str: str) -> bool:
    """
    Validate whether the response follows the format:
    <think>...</think><answer>...</answer>,
    ensuring all tags appear exactly once and in the correct order.
    """
    validation_passed = True
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

    # Ensure correct tag ordering
    if not validation_passed or (
        positions['think_start'] == -1 or
        positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        validation_passed = False

    return validation_passed


def remove_prefix(text: str) -> str:
    """
    Remove multi-turn chat prefixes such as
    <|im_start|>system...<|im_start|>assistant from the input text.
    """
    pattern = re.compile(
        r'<\|im_start\|>system.*?<\|im_start\|>user.*?<\|im_start\|>assistant',
        re.DOTALL
    )
    try:
        cleaned_text = pattern.sub('', text)
        return cleaned_text.strip()
    except re.error:
        return text


# --- Main Scoring Function (Modified to accept raw_log_line) ---

def compute_score(solution_str: str, 
                  ground_truth_template: str,
                  raw_log_line: str,
                  format_reward: float = 0.1,
                  answer_reward: float = 1.0) -> float:
    """
    Compute a composite score for the model output,
    including a structure validation reward and a content (F1) reward.
    """
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))

    # Remove conversation prefix
    solution_str = remove_prefix(solution_str)

    # Extract model reasoning and answer
    think_text, answer_text = extract_solution(solution_str)

    # 1. Validate response structure
    format_correct = validate_response_structure(solution_str)
    format_score = format_reward if format_correct else -abs(format_reward * 10)
    print(f"\n Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f" Format score: {format_score:.2f}")

    # 2. Evaluate answer content using F1 score
    answer_score = 0.0
    if format_correct and answer_text is not None:
        print(f"\n[Content Validation (F1)]")
        print(f" Raw Log: {raw_log_line}")
        print(f" Expected Template: {ground_truth_template}")
        print(f" Predicted Answer (Template): {answer_text}")

        labels = [ground_truth_template]
        predicts = [solution_str]
        raw_logs = [raw_log_line]

        f1 = f1_reward(labels, predicts, raw_logs)
        answer_score = f1 * answer_reward

        print(f" F1 Score (Template/Variable match): {f1:.4f}")
    else:
        answer_score = -1.0
        print("\n[Content Validation] Skipped due to format errors or missing answer.")

    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f" Format: {format_score:.2f}")
    print(f" Answer: {answer_score:.2f}")
    print(f" Total: {total_score:.2f}")
    print("="*80 + "\n")

    return total_score
