import os
from datasets import Dataset, load_dataset
# Assuming 'verl.utils.hdfs_io' and 'qwen_chat_templater' are available in the execution environment
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from qwen_chat_templater import PromptTemplate as PT  # PT is an alias for the PromptTemplate class
import pandas as pd
import pyarrow.parquet as pq

def make_prefix(dp, template_type):
    """
    Constructs the prompt prefix based on the data point and template type.
    
    The prompt is designed to instruct the model to use Chain-of-Thought 
    (<think>) and structured answer (<answer>) tags.
    """
    quiz = dp['instruction'] + '\n' + dp['input']
    
    if template_type == 'base':
        prefix = f"""
        The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process \
        in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within \
        <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> \
        <answer> answer here </answer>. Now the user asks you to solve a log parsing reasoning problem. After thinking, \
        when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. \
        A log template only retains static parts in a log and replaces dynamic variables with a symbol of <*>. \
        for example, <answer>CE sym <*>, at <*>, mask <*></answer>.\n\nUser:{quiz}\nAssistant: <think>
        """
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""

    elif template_type == 'qwen-logIRS':
        pt = PT(system_prompt="The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Now the user asks you to solve a log analysing reasoning problem. After thinking, when you finally reach a conclusion, clearly provide the answer within <answer> </answer> tags.")
        pt.add_user_message(quiz)
        prefix = pt.build_prompt()

        # Induce <think> tag for Chain-of-Thought reasoning
        prefix += "<think>"
    elif template_type == 'qwen-logAnomaly':
        pt = PT(system_prompt="The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Now the user asks you to solve a log anomaly detection problem. After thinking, when you finally reach a conclusion, clearly provide the answer within <answer> </answer> tags. Your answer must be one of two cases: normal or abnormal.")
        pt.add_user_message(quiz)
        prefix = pt.build_prompt()

        # Induce <think> tag for Chain-of-Thought reasoning
        prefix += "<think>"
    elif template_type == 'verl-qwen-chat-logParsing':
        prefix = "<|im_start|>system\nThe user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Now the user asks you to solve a log analysing reasoning problem. After thinking, when you finally reach a conclusion, clearly provide the answer within <answer> </answer> tags.\n<|im_end|>"
        prefix += quiz
    elif template_type == 'verl-qwen-chat-logIRS':
        prefix = "<|im_start|>system\nThe user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Now the user asks you to solve a log-analysing reasoning problem. After thinking, when you finally reach a conclusion, clearly provide the answer within <answer> </answer> tags.\n<|im_end|>"
        prefix += quiz

    return prefix

def preprocess_dataset(
    input_file: str,
    output_dir: str,
    data_source: str,
    template_type: str = "verl-qwen-chat-logIRS",
    hdfs_dir: str = None,
    split: str = "train"
):
    """
    Preprocesses the dataset and saves it in a specified format (Parquet and CSV).
    
    Args:
        input_file: Path to the input JSON file.
        output_dir: Path to the local output directory.
        data_source: Identifier for the data source (e.g., 'logIRS').
        template_type: The prompt template type to apply (e.g., 'llama-2-chat-logParsing').
        hdfs_dir: Optional HDFS output directory for copying results.
        split: Dataset split identifier ('train' or 'test').
    """
    
    # Create local output directory
    os.makedirs(output_dir, exist_ok=True)

    # Data generator function
    def gen_from_json(path):
        """Loads data points from a JSON file."""
        with open(path) as f:
            dataset = json.load(f)
        print(f'Loaded dataset: {len(dataset)} samples')
        for item in dataset:
            yield item

    # Mapping function for dataset processing
    def make_map_fn():
        """Returns the function to process individual examples."""
        def process_fn(example, idx):
            question = make_prefix(example, template_type=template_type)
            
            # Check for excessive prompt length
            if len(question) >= 2048:
                print("Long question detected:", question)

            solution = example['output']
            
            return {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "log",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
        return process_fn

    # Generate dataset from generator
    base_name = os.path.basename(input_file).split('.')[0]
    dataset = Dataset.from_generator(
        gen_from_json, 
        gen_kwargs={'path': input_file}
    )

    # Process the dataset using the map function
    processed_ds = dataset.map(
        function=make_map_fn(),
        with_indices=True
    )

    # Save in Parquet format
    parquet_path = os.path.join(output_dir, f'{base_name}_{template_type}.parquet')
    processed_ds.to_parquet(parquet_path)

    # Convert Parquet to CSV
    csv_path = os.path.join(output_dir, f'{base_name}_{template_type}.csv')
    df = pq.read_table(parquet_path).to_pandas()
    df.to_csv(csv_path, index=False)

if __name__ == '__main__':

    train_file_path = 'path_to_reasoning'
    parquet_dir = './parquet_data'
    data_source = 'logParsing/logAnomaly/logIRS'
    template_type = ''

    preprocess_dataset(
        input_file=train_file_path,
        output_dir=parquet_dir,
        data_source=data_source,
        template_type=template_type,
        split='train'
    )
