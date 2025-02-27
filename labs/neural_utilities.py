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

from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI

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