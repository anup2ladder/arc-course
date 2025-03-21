import re
import itertools
import math
import time
import random
from collections import Counter
from numpy.random import choice, randint
import numpy as np
from IPython.display import HTML, display, clear_output
import matplotlib.pyplot as plt
import ipywidgets as widgets
import itertools

# NOTE: PatchFastRL needs to run **before** the imports below
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import torch, gc
from datasets import load_dataset, Dataset
from transformers import EarlyStoppingCallback, TextStreamer, TrainingArguments
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
from unsloth.chat_templates import get_chat_template
from vllm import SamplingParams

from symbolic_utilities import ltgrammar, lt_nonterminals, lt_terminals, lt_eval_dict

from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI

def interpret(string, eval_dict):
    # in case I decide to do something fancy later
    return eval(string, eval_dict)

def levenshtein_distance(s1, s2):
    # Create a matrix of zeros
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    # Initialize the matrix
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Compute the dp matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],    # deletion
                                   dp[i][j - 1],    # insertion
                                   dp[i - 1][j - 1])  # substitution
    return dp[m][n]

def normalized_utility(predicted, ground_truth):
    """
    Compute a utility value based on the normalized edit distance.
    
    Parameters:
    - predicted: list of predicted integers.
    - ground_truth: list of ground truth integers.
    
    Returns:
    - Utility value between 0 and 1.
    """
    d = levenshtein_distance(predicted, ground_truth)
    max_len = max(len(predicted), len(ground_truth))
    # Avoid division by zero when both sequences are empty
    if max_len == 0:
        return 1.0
    normalized_d = d / max_len
    utility = 1 - normalized_d
    return utility

###### Visualization utils

def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

####### Neural program synthesis
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_xml_reasoning(text: str) -> str:
    reasoning = text.split("<reasoning>")[-1]
    reasoning = reasoning.split("</reasoning>")[0]
    return reasoning.strip()

def produce_tasks(eval_dict, sentences_pool=None, n_inputs=3, n_tasks=10):
    
    if sentences_pool is None:
        sentences = []
        for i in range(n_tasks):
            random_sentence = "".join(tree_to_sentence(generate_tree("T", ltgrammar)))
            random_f = interpret(random_sentence, eval_dict)
            while random_sentence in sentences:
                random_sentence = "".join(tree_to_sentence(generate_tree("T", ltgrammar)))
            sentences.append(random_sentence)
    else:
        sentences = choice(sentences_pool, n_tasks, replace=False)
    
    tasks = []
    for i, sent in enumerate(sentences):
        inputs = [
            list(randint(0, 10, randint(2, 6)))
            for _ in range(n_inputs)
        ]
        random_f = interpret(sent, eval_dict)
        outputs = [
            random_f(i) 
            for i in inputs
        ]
        examples = list(zip(inputs,outputs))
        tasks.append({
            'sentence': sent,
            'examples': examples,
            'task': "\n".join([f"-{i} -> {o}" for i, o in examples])
        })
    return tasks

def get_data(grammar, system_prompt, **kwargs):
    data = produce_tasks(**kwargs)
    data = Dataset.from_list(data)
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': x['task']}
        ]
    }) 
    return data 

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        # count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        # count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def is_in_lt_CFG(string):
    # Build a regex pattern. re.escape ensures that any special characters in substrings are treated literally.
    pattern = re.compile('^(?:' + '|'.join(re.escape(s) for s in lt_terminals) + ')+$')  
    return pattern.fullmatch(string)

def print_func(prompts, completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*30, f"\n**Question**\n{q}", f"\n\n**Response**\n{responses[0]}", f"\n\n**Extracted**\n{extracted_responses[0]}\n\n")
    return 0

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def cfg_reward_func(completions, lt_terminals, **kwargs):
    """
    Check whether the expression belongs to the cfg
    """
    responses = [
        completion[0]['content'] 
        for completion in completions
    ]
    answers = [
        extract_xml_answer(x) 
        for x in responses
    ]
    return [
        0.5 if is_in_lt_CFG(answer) else 0.0
        for answer 
        in answers
    ]

def lt_correctness_reward_func(prompts, completions, examples, **kwargs) -> list[float]:
    """
    Whether the answer is 
        - evaluable into a function
        - evaluates into the *right* function
    """
    utilities = [0]*len(completions)
    
    responses = [
        completion[0]['content'] 
        for completion in completions
    ]
    answers = [
        extract_xml_answer(x) 
        for x in responses
    ]

    # Check if code runs
    for i, answer in enumerate(answers):
        if is_in_lt_CFG(answer):
            try:
                f = eval(answer, lt_eval_dict)    
                # check predictions
                for (inp, out) in ex:
                    print("Prediction: ", answers[i]) 
                    print(f(inp), "->", out)
                    utilities[i] += normalized_utility(out, f(inp))
            except:
                pass
    
    return utilities

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has specific format (with newlines)."""
    pattern = r"(^|\n)<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n($|\n)"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if the completion has a specific format 
    (don't care about newlines or starting and ending with blocks).
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


##### without reasoning

def direct_cfg_reward_func(completions, **kwargs):
    """
    Check whether the expression belongs to the cfg
    """
    responses = [
        completion[0]['content'] 
        for completion in completions
    ]
    return [
        0.5 if is_in_lt_CFG(response) else 0.0
        for response in responses
    ]

def direct_lt_correctness_reward_func(prompts, completions, examples, **kwargs) -> list[float]:
    """
    Whether the answer is 
        - evaluable into a function
        - evaluates into the *right* function
    """
    
    utilities = [0]*len(completions)
    responses = [
        completion[0]['content'] 
        for completion in completions
    ]
    # Check if code runs
    for i, answer in enumerate(responses):
        if is_in_lt_CFG(answer):
            try:
                f = eval(answer, lt_eval_dict)
                # check predictions
                for (inp, out) in examples[i]:
                    print("Prediction: ", responses[i]) 
                    print(inp, "-->", f(inp), "vs", out)
                    utilities[i] += normalized_utility(out, f(inp))
            except:
                pass
        else:
            print(f"{answer} not in cfg")
    return utilities

def direct_conciseness_reward_func(prompts, completions, examples, **kwargs) -> list[float]:
    """
    This is meant to make the model avoid redundant operations.
    However, be careful not to make it bigger than the accuracy reward,
    otherwise the model will just end up generating empty strings!
    """
    utilities = [0]*len(completions)
    responses = [
        completion[0]['content'] 
        for completion in completions
    ]
    # Check if code runs
    for i, answer in enumerate(responses):
        bonus = 0
        # add a monotonically decreasing function of length
        # for correct generations.
        # This penalizes length while making sure correct programs have stricly
        # higher utility than incorrect ones
        if is_in_lt_CFG(answer):
            try:
                f = eval(answer, lt_eval_dict)
                # if predictions are perfect, try to minimize length of formula
                if all([f(inp)==out for inp, out in examples[i]]):
                    # this is higher for shorter generations
                    utilities[i] = np.exp(-len(answer)*0.01)
            except Exception:
                pass
        else:
            # function was not in CFG
            print(f"{answer} not in cfg")    
    return utilities