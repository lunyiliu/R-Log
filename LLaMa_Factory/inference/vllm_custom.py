# -*- coding: UTF-8 -*-
"""
@Project : Llama-Factory
@File    : api.py
@Author  : young
@Date    : 2024/10/16 14:50
@Desc    : Script for running VLLM-based inference on specified models and datasets.
"""

import gc
import json
import os
import sys
import time
import traceback
import argparse
import multiprocessing as mp
import torch
from vllm import LLM, SamplingParams
from qwen_chat_templater import PromptTemplate as PT


# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Run inference using a VLLM-based model.")
parser.add_argument('--model', '-m', type=str, required=True,
                    help="Model name defined in models_config.json")
parser.add_argument('--testset', '-t', type=str, required=True,
                    help="Testset name defined in testsets_config.json")

args = parser.parse_args()


# -------------------------------
# Utility Functions
# -------------------------------
def load_config(file_path: str) -> dict:
    """
    Load a JSON configuration file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# -------------------------------
# Prompt Construction
# -------------------------------
def make_prefix(dp, template_type):
    """
    Build the task prefix (prompt) for a given dataset entry and template type.
    """
    quiz = dp['instruction'] + '\n' + dp['input']

    if template_type == 'base':
        prefix = f"""
        The user asks a question, and the Assistant solves it. The assistant first reasons internally 
        before giving the final answer. The reasoning and answer are enclosed in <think></think> and 
        <answer></answer> tags respectively, e.g.:
        <think> reasoning process here </think> <answer> answer here </answer>.
        The user now asks a log parsing reasoning question. Provide the answer clearly within 
        <answer></answer> tags. A log template retains only static parts and replaces dynamic variables 
        with <*> (e.g., <answer>CE sym <*>, at <*>, mask <*></answer>).

        User: {quiz}
        Assistant: <think>
        """

    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are a helpful assistant. You first think through the reasoning internally before providing the answer.
Enclose reasoning and answer in <think></think> and <answer></answer> tags respectively.
Now solve the given logical reasoning problem, and when done, present each identified entity inside 
<answer></answer> tags (e.g., <answer>(1) Zoey is a knight\n(2) ...</answer>).
<|im_end|>
<|im_start|>user
{quiz}
<|im_end|>
<|im_start|>assistant
<think>"""

    elif template_type == 'llama-2-chat-logParsing':
        pt = PT(system_prompt=(
            "The user asks a question, and the Assistant solves it. The assistant first thinks internally "
            "and then provides the final answer enclosed in <think></think> and <answer></answer> tags. "
            "Now the user asks a log parsing reasoning question. A log template retains static parts and "
            "replaces dynamic variables with <*>. Provide only the templated logs within <answer></answer>."
        ))
        pt.add_user_message(quiz)
        prefix = pt.build_prompt() + "<think>"

    elif template_type in ['llama-2-chat-IRS', 'qwen-logIRS']:
        pt = PT(system_prompt=(
            "The user asks a question, and the Assistant solves it. The assistant reasons first and then provides "
            "the final answer enclosed in <think></think> and <answer></answer>. Now solve the given log analysis "
            "reasoning problem and present your conclusion clearly within <answer></answer> tags."
        ))
        pt.add_user_message(quiz)
        prefix = pt.build_prompt() + "<think>"

    elif template_type == 'qwen-logAnomaly':
        pt = PT(system_prompt=(
            "The user asks a question, and the Assistant solves it. The assistant reasons internally and then provides "
            "the final answer enclosed in <think></think> and <answer></answer> tags. Solve the log anomaly detection "
            "problem and clearly output either 'normal' or 'abnormal' within <answer></answer> tags."
        ))
        pt.add_user_message(quiz)
        prefix = pt.build_prompt() + "<think>"

    elif template_type in ['verl-qwen-chat-logParsing', 'verl-qwen-chat-logIRS']:
        prefix = (
            "<|im_start|>system\nThe user asks a question, and the Assistant solves it. "
            "The reasoning and answer are enclosed within <think></think> and <answer></answer> tags respectively. "
            "Now solve the log analysis reasoning problem.\n<|im_end|>" + quiz
        )

    return prefix


# -------------------------------
# Prompt Templates
# -------------------------------
def default_template(prompt):
    return f"Human: {prompt}\nAssistant:"


def llama_2_template(prompt):
    pt = PT()
    pt.add_user_message(prompt)
    return pt.build_prompt()


def qwen_logAnomaly_template(prompt):
    pt = PT(system_prompt=(
        "You are solving a log anomaly detection task. After reasoning internally, "
        "output the result within <answer></answer> tags — either 'normal' or 'abnormal'."
    ))
    pt.add_user_message(prompt)
    return pt.build_prompt() + "<think>"


def qwen_logParsing_template(prompt):
    pt = PT(system_prompt=(
        "You are solving a log parsing reasoning task. A log template keeps only static parts "
        "and replaces dynamic variables with <*>. Provide only the final log template inside <answer></answer>."
    ))
    pt.add_user_message(prompt)
    return pt.build_prompt() + "<think>"


def qwen_logIRS_template(prompt):
    pt = PT(system_prompt=(
        "You are solving a log analysis reasoning task. Think internally and provide your conclusion "
        "within <answer></answer> tags."
    ))
    pt.add_user_message(prompt)
    return pt.build_prompt() + "<think>"


# -------------------------------
# Inference Function
# -------------------------------
def vllm_infer(model_path, dataset_path, output_file, infer_template_fun):
    """
    Run inference with a given model and dataset using vLLM.

    Args:
        model_path (str): Path to the model.
        dataset_path (str): Path to the dataset JSON.
        output_file (str): Path to save the output results.
        infer_template_fun (callable): Function to build inference prompts.
    """
    sampling_params = SamplingParams(temperature=0.95, top_p=0.7, max_tokens=2048)
    prompts, results = [], []

    llm = LLM(
        model=model_path,
        max_model_len=4096,
        trust_remote_code=True,
        tokenizer=model_path,
        tokenizer_mode='auto',
        gpu_memory_utilization=0.9,
        tensor_parallel_size=4
    )

    with open(dataset_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
        items = items[:1]  # ⚠️ Limit for testing; remove for full run
        print(f"Loaded {len(items)} items from dataset.")
        print(f"First item sample: {items[0]}")

        for i, item in enumerate(items):
            raw_prompt = item["instruction"] + "\n" + item["input"]
            print(f"[{i}] Original prompt:\n{raw_prompt}")
            templated_prompt = infer_template_fun(raw_prompt)
            print(f"[{i}] Templated prompt:\n{templated_prompt}")
            prompts.append(templated_prompt)

    try:
        print("Starting VLLM generation...")
        outputs = llm.generate(prompts, sampling_params)
        print("VLLM generation completed successfully.")

        for i, output in enumerate(outputs):
            prediction = output.outputs[0].text
            print(f"[{i}] Prediction:\n{prediction}")
            results.append({
                "instruction": items[i]["instruction"],
                "input": items[i]["input"],
                "output": prediction
            })

            if len(results) % 100 == 0:
                print(f"Checkpoint: Saving {len(results)} results...")
                with open(output_file, 'w', encoding='utf-8') as f_out:
                    json.dump(results, f_out, ensure_ascii=False, indent=4)

        print("Final save in progress...")
        json.dump(results, open(output_file, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=4)
        print("Inference completed and saved successfully.")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print(parse_exception_traceback(e))

    finally:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
        print("Cleanup finished.")


# -------------------------------
# Evaluation Wrapper
# -------------------------------
def eval_infer(model_name, testset_name):
    """Load configurations, prepare paths, and launch inference."""
    models_config = load_config('models_config.json')
    testsets_config = load_config('testsets_config.json')

    try:
        model_path = models_config[model_name]
        testset_info = testsets_config[testset_name]
        dataset_path = testset_info['path']
        template_type = testset_info['type']
    except KeyError as e:
        print(f"Error: '{e.args[0]}' not found in configuration files.")
        return

    function_map = {
        "parsing": qwen_logParsing_template,
        "irs": qwen_logIRS_template,
        "anomaly": qwen_logAnomaly_template
    }
    infer_template_fun = function_map.get(template_type)
    if not infer_template_fun:
        print(f"Error: No template function found for '{template_type}'.")
        return

    output_dir = './LLaMA-Factory/original_vllm_output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{testset_name}_by_{model_name}.json'

    print(f"Running inference using model '{model_name}' on testset '{testset_name}'...")
    model_path = ""
    vllm_infer(model_path, dataset_path, output_file, infer_template_fun)
    print("✅ Inference finished successfully!")


def parse_exception_traceback(exception):
    """Return formatted traceback string for better readability."""
    exc_type, exc_value, exc_tb = sys.exc_info()
    trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    return (f"Exception Type: {type(exception)}\n"
            f"Message: {exception}\nTraceback:\n\n{trace}")


# -------------------------------
# Main
# -------------------------------
if __name__ == '__main__':
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    mp.set_start_method('spawn')
    eval_infer(args.model, args.testset)
