import re
import math
from collections import Counter
from sklearn.metrics import f1_score
from transformers import AutoTokenizer # Note: Requires 'transformers' to be installed

def merge_subwords(subwords: list) -> list:
    """Merges subwords that were split by a tokenizer (e.g., '##word')."""
    merged_words = []
    for subword in subwords:
        if "##" in subword and merged_words:
            # Append to the previous word if '##' is present
            merged_words[-1] += subword.replace("##", "")
        else:
            merged_words.append(subword)
    return merged_words

def filter_special_chars_for_F1(s: list) -> list:
    """Filters out special characters from a list of tokens for F1 calculation."""
    # Define special characters to filter out (anything that is not a word char, whitespace, or '*')
    special_chars = r'[^\w\s*]'
    
    # Replace all special characters with an empty string for each element
    filtered_str = [re.sub(special_chars, '', ele) for ele in s]
    
    # Remove any elements that became empty strings after filtering
    filtered_str = [ele for ele in filtered_str if ele != '']
    return filtered_str

def extract_variables(template: str, raw_log: str) -> list:
    """
    Extracts variable values from a raw log based on a provided template.
    
    The template uses '<#>' as a placeholder for variables. It also handles 
    replacing common regex characters that are meant to be literals.
    """
    # Replace '*' placeholders with a temporary symbol for unified handling
    raw_log = raw_log.replace('*', '#')
    template = template.replace('*', '#')
    
    # Escape regex special characters in the template
    template = template.replace('(', '\(')
    template = template.replace(')', '\)')
    template = template.replace('[', '\[')
    template = template.replace(']', '\]')
    
    # Clean up redundant variable placeholders (e.g., '<#><#>' becomes '<#>')
    template = re.sub(r"(<#>\s?)+", "<#>", template)
    
    # Convert template placeholder '<#>' to the regex capture group (.*)
    pattern = template.replace('<#>', '(.*)')

    # Search for matches in the raw log
    matches = re.findall(pattern, raw_log)

    if matches:
        # If matches are found, convert the result to a list of stripped strings
        if isinstance(matches[0], str):
            return [matches[0].strip()]
        else:
            # Handles multiple capture groups
            return [match.strip() for match in matches[0]]
    else:
        return []

def f1_reward(true_labels: list, predicted_labels: list, raw_logs: list) -> float:
    """
    Calculates the token-level F1 score for variable identification.
    
    This function tokenizes both true and predicted templates and determines 
    if each token is a fixed template word or a variable, then calculates F1.
    """
    # NOTE: Hardcoded tokenizer path maintained from original code.
    tokenizer = AutoTokenizer.from_pretrained("/data/wx1397120/bert-base-cased") 
    
    true_tokens = []
    predicted_tokens = []
    
    # 1. Extract variables from raw logs based on true template
    variables_ls = []
    for template, log in zip(true_labels, raw_logs):
        variables = extract_variables(template, log)
        variables_ls.append(variables)

    # 2. Process each pair of true/predicted template
    F1_macro = []
    for true, pred, variables in zip(true_labels, predicted_labels, variables_ls):
        
        if not isinstance(pred, str):
            # Skip non-string predictions, record a negative F1 score
            F1_macro.append(-1)
            continue
        
        # Normalize true template: replace '<*>' with '<#>' for cleaning, then back
        true = true.replace('<*>', '<#>').replace('*', '#')
        true = re.sub(r"(<#>\s?)+", "<#>", true)
        true = true.replace('<#>', '<*>')
        
        # Tokenize true template and filter special chars
        true_token = filter_special_chars_for_F1(merge_subwords(tokenizer.tokenize(true)))
        
        # Tokenize predicted template and filter special chars
        # pred_tmp is used to track variable matching within the log content
        pred_tmp = pred 
        # Remove *variable* markers if present, as they are not used for token matching
        pred_tmp = re.sub(r'\*([^*\s]+)\*', "", pred_tmp) 
        predict_raw = filter_special_chars_for_F1(merge_subwords(tokenizer.tokenize(pred)))
        predict_raw = [ele.lower() for ele in predict_raw]
        
        predict_token = []
        variable_index = 0

        # Align predicted tokens with true tokens
        for token in true_token:
            if token == '*':
                # Case 1: True token is a variable marker '*'
                if variable_index < len(variables):
                    current_variable = variables[variable_index]
                    # Check if the variable content is still present in the predicted template (i.e., it was NOT recognized as a variable)
                    # or if the predicted template contains no variable markers ('*')
                    if current_variable in pred_tmp or '*' not in pred:
                        predict_token.append('template') # Variable incorrectly treated as template content
                        # Remove matched variable content from the temporary prediction string
                        pred_tmp = pred_tmp.replace(current_variable, '', 1)
                    else:
                        predict_token.append('*') # Variable correctly recognized as a variable
                    variable_index += 1
                else:
                    predict_token.append('*') # Fallback if variable list is exhausted (should not happen often)
            else:
                # Case 2: True token is a template word
                if token.lower() in predict_raw:
                    predict_token.append(token) # Template word correctly predicted
                    predict_raw.remove(token.lower()) # Consume the token
                else:
                    predict_token.append('*') # Template word incorrectly treated as a variable
                    
        # Calculate F1 for this single instance
        true_class_macro = ["variable" if token == "*" else "template" for token in true_token]
        predicted_class_macro = ["variable" if token == "*" else "template" for token in predict_token]
        
        if true_class_macro:
             F1_macro.append(f1_score(true_class_macro, predicted_class_macro, pos_label="variable", zero_division=0))
        
        # Global lists for micro F1
        true_tokens.extend(true_token)
        predicted_tokens.extend(predict_token)

    # Calculate overall micro F1 score
    true_classes = ["variable" if token == "*" else "template" for token in true_tokens]
    predicted_classes = ["variable" if token == "*" else "template" for token in predicted_tokens]

    f1 = f1_score(true_classes, predicted_classes, pos_label="variable", zero_division=0)
    return f1

def extract_solution(solution_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts the final thought (<think>) and answer (<answer>) from the model's response string.
    
    Returns:
        Tuple[think_text, answer_text]
    """
    # Regex to match <think> and <answer> tags and their content
    think_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'

    # re.DOTALL makes '.' match newlines
    think_match = re.search(think_pattern, solution_str, re.DOTALL)
    answer_match = re.search(answer_pattern, solution_str, re.DOTALL)

    final_think = think_match.group(1).strip() if think_match else None
    final_answer = answer_match.group(1).strip() if answer_match else None
    
    # Note: Error printing removed as per optimization request.
    
    return final_think, final_answer

def remove_prefix(text: str) -> str:
    """
    Removes common multi-line chat prefix patterns used by some models.
    
    The function matches and removes the content typically found before the 
    actual model output, often spanning 'system', 'user', and 'assistant' roles 
    within markers like '<|im_start|>' and '[INST]'.
    """
    # Pattern to match the full conversation history prefix up to '<|im_start|>assistant'
    pattern = re.compile(
        r'<\|im_start\|>system.*?<\|im_start\|>user.*?<\|im_start\|>assistant',
        re.DOTALL
    )
    
    try:
        # Substitute the pattern with an empty string
        cleaned_text = pattern.sub('', text)

        if cleaned_text != text:
            # If a substitution occurred, strip leading/trailing whitespace
            return cleaned_text.strip()
        else:
            # If no match/change, return the original text
            return text
            
    except re.error:
        # If the regex itself is malformed, return original text
        return text

def validate_response_structure(processed_str: str) -> bool:
    """
    Performs comprehensive validation of response structure for <think>...</think><answer>...</answer> format.
    
    Checks for tag count (exactly one of each) and correct tag order.
    """
    validation_passed = True

    # Check required tags and counts
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
            # print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order: <think>...</think><answer>...</answer>
    if (positions.get('think_start', -1) > positions.get('think_end', -1) or
        positions.get('think_end', -1) > positions.get('answer_start', -1) or
        positions.get('answer_start', -1) > positions.get('answer_end', -1)):
        # print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False

    return validation_passed

# --- Metrics Calculation Functions (Moved from bottom for better structure) ---

def calculate_ngram_counts(tokens: list, n: int) -> Counter:
    """Calculates the counts of all n-grams in a list of tokens."""
    counts = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        counts[ngram] += 1
    return counts

def get_bleu_score(reference: str, hypothesis: str) -> float:
    """Calculates the BLEU-4 score (modified precision with brevity penalty)."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # Calculate brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len == 0:
        return 0.0
        
    brevity_penalty = min(1.0, math.exp(1 - ref_len / hyp_len)) if hyp_len < ref_len else 1.0

    # Calculate modified n-gram precision
    weights = [0.25, 0.25, 0.25, 0.25]
    p_n = [0.0] * 4
    for n in range(1, 5):
        hyp_ngrams = calculate_ngram_counts(hyp_tokens, n)
        ref_ngrams = calculate_ngram_counts(ref_tokens, n)

        clipped_count = 0
        total_count = sum(hyp_ngrams.values())
        if total_count == 0:
            continue
        
        for ngram, count in hyp_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))
        
        p_n[n-1] = clipped_count / total_count
        
    # Combine the precision scores
    score = 0.0
    for i, p in enumerate(p_n):
        if p > 0:
            score += weights[i] * math.log(p)
        else:
            return 0.0 # If any precision is zero, the final score is zero

    return brevity_penalty * math.exp(score)

def get_rouge_score(reference: str, hypothesis: str) -> dict:
    """Calculates the ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)

    if hyp_len == 0 or ref_len == 0:
        return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}

    def calculate_f1(matches: int, total_pred: int, total_ref: int) -> float:
        """Helper to calculate F1 score from match counts."""
        precision = matches / total_pred if total_pred > 0 else 0
        recall = matches / total_ref if total_ref > 0 else 0
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    # ROUGE-1 (Unigram overlap)
    ref_counts_1 = Counter(ref_tokens)
    hyp_counts_1 = Counter(hyp_tokens)
    matches_1 = sum(min(hyp_counts_1[token], ref_counts_1[token]) for token in hyp_counts_1)
    rouge_1_f = calculate_f1(matches_1, hyp_len, ref_len)

    # ROUGE-2 (Bigram overlap)
    ref_counts_2 = calculate_ngram_counts(ref_tokens, 2)
    hyp_counts_2 = calculate_ngram_counts(hyp_tokens, 2)
    matches_2 = sum(min(hyp_counts_2[ngram], ref_counts_2[ngram]) for ngram in hyp_counts_2)
    # The total number of bigrams is (length - 1)
    rouge_2_f = calculate_f1(matches_2, max(0, hyp_len - 1), max(0, ref_len - 1))

    # ROUGE-L (Longest Common Subsequence - LCS)
    # Dynamic Programming to find LCS length
    dp = [[0] * (ref_len + 1) for _ in range(hyp_len + 1)]
    for i in range(1, hyp_len + 1):
        for j in range(1, ref_len + 1):
            if hyp_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[hyp_len][ref_len]
    rouge_l_f = calculate_f1(lcs_len, hyp_len, ref_len)
    
    return {
        'rouge-1': {'f': rouge_1_f}, 
        'rouge-2': {'f': rouge_2_f}, 
        'rouge-l': {'f': rouge_l_f}
    }

def get_scores(ref: str, pred: str) -> list:
    """Calculates and returns BLEU-4 and ROUGE-1/2/L scores (as percentages)."""
    # Calculate BLEU-4 score
    bleu_score = get_bleu_score(ref, pred)
    BLEU = 100 * bleu_score
    # print(f"BLEU-4 Score: {BLEU:.2f}") # Printing removed

    # Calculate ROUGE scores
    scores = get_rouge_score(ref, pred)
    ROUGE_1 = 100 * scores["rouge-1"]['f']
    ROUGE_2 = 100 * scores["rouge-2"]['f']
    ROUGE_l = 100 * scores["rouge-l"]['f']
    # Printing removed
    
    return [BLEU, ROUGE_1, ROUGE_2, ROUGE_l]

def compute_score(solution_str: str, 
                  ground_truth: str,
                  format_reward: float = 0.1,
                  answer_reward: float = 1.0) -> float:
    """
    Computes a comprehensive score for a model response based on format and content similarity.
    
    Args:
        solution_str: Raw model response string.
        ground_truth: Ground truth string (used as the reference for metrics).
        format_reward: Base score for correct format.
        answer_reward: Weight factor for answer content score (unused in final calculation logic, but kept as a parameter).
        
    Returns:
        Total score (sum of format and answer rewards).
    """

    # 1. Preprocess and Extract Model Answer
    solution_str = remove_prefix(solution_str)
    think_text, answer_text = extract_solution(solution_str)

    # 2. Validate response structure
    format_correct = validate_response_structure(solution_str)
    
    # Assign format score: large penalty for incorrect format
    format_score = format_reward if format_correct else -abs(format_reward * 10)
    # print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}") # Printing removed
    # print(f"  Format score: {format_score}") # Printing removed

    # 3. Validate answer content
    answer_score = 0.0
    if format_correct and answer_text:
        # print(f"\n[Content Validation]") # Printing removed
        # print(f"  Expected: {ground_truth}") # Printing removed
        # print(f"  Predicted: {answer_text}") # Printing removed
        
        # Calculate scores (BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L)
        scores = get_scores(ground_truth, answer_text)
        
        # Calculate average of the four scores (normalized to 1.0)
        answer_score = sum(scores) / 400
        
        # Apply a modification factor (0.8 maintained from original logic)
        if answer_score > 0:
            answer_score = answer_score / 0.8
        else:
            answer_score = 0.0

    else:
        # Penalize content score if format is incorrect or answer is missing
        answer_score = -1.0
        # Printing removed

    # 4. Final Score Calculation
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
