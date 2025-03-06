import sys
sys.path.append("..")#

import re
import itertools
import math
import time
import random
from collections import Counter
from pprint import pprint

import numpy as np
import pandas as pd
from numpy.random import choice, randint

from symbolic_utilities import enumerate_full_sentences, define_lt_DSL

from neural_utilities import \
    extract_xml_answer, extract_xml_reasoning, produce_tasks, get_data, \
    print_func, lt_correctness_reward_func, \
    xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func, \
    direct_cfg_reward_func, direct_lt_correctness_reward_func

# NOTE: PatchFastRL needs to run **before** the imports below
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import torch, gc
from datasets import load_dataset, Dataset
from transformers import EarlyStoppingCallback, TextStreamer, TrainingArguments
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
from unsloth.chat_templates import get_chat_template
from vllm import SamplingParams

if __name__=="__main__":
    
    max_seq_length = 1024 # Can increase for longer reasoning traces
    lora_rank = 64 # Larger rank = smarter, but slower

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

    ltgrammar, lt_nonterminals, lt_terminals, lt_eval_dict = define_lt_DSL()

    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name = "Qwen/Qwen2.5-0.5B-Instruct",
        model_name="unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        # False for LoRA 16bit
        load_in_4bit=True, 
        # Enable vLLM fast inference
        fast_inference=True, 
        max_lora_rank=lora_rank,
        # Reduce if out of memory
        gpu_memory_utilization=0.5, 
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        r=lora_rank, 
        # Which parts of the model are we gonna train?
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj",
            "gate_proj", 
            "up_proj", 
            "down_proj",
        ], 
        lora_alpha = lora_rank,
        # Enable long context finetuning
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    lt_system_prompt = ""

    # get 5000 sentences
    sentences_pool = []
    for i, sent in enumerate(enumerate_full_sentences('T', ltgrammar, max_depth=5)):
        if i==5000:
            break
        sentences_pool.append(sent)
    
    data = get_data(
        ltgrammar, 
        lt_system_prompt, 
        eval_dict=lt_eval_dict, 
        n_tasks=5000, 
        sentences_pool=sentences_pool
    )

    data = data.map(lambda x: {
        'completion': [{'content': x['sentence'], 'role': 'assistant'}], 
        'lt_terminals': lt_terminals
    })

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=3e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=10,
            output_dir="SLURM_lt_direct_output",
            seed=0,
            save_steps=40
        ),
    )
    
    trainer.train()

    model.save_lora('SLURM_finetuned_lt')

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "SLURM_finetuned_lt",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        print_func,
        direct_cfg_reward_func,
        direct_lt_correctness_reward_func
    ],
    args=GRPOConfig(
            # use vLLM for fast inference!
            use_vllm = True, 
            learning_rate = 5e-6,
            adam_beta1 = 0.9,
            adam_beta2 = 0.99,
            weight_decay = 0.1,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            optim = "adamw_8bit",
            logging_steps = 1,
            bf16 = is_bfloat16_supported(),
            fp16 = not is_bfloat16_supported(),
            per_device_train_batch_size = 1,
            # Increase to 4 for smoother training
            gradient_accumulation_steps = 1, 
            # Decrease if out of memory
            num_generations = 8, 
            max_prompt_length = 256,
            max_completion_length = 32,
            # Set to 1 for a full training run
            num_train_epochs = 1, 
            max_steps = 500,
            save_steps = 50,
            max_grad_norm = 0.1,
            # Can use Weights & Biases
            report_to = "none", 
            output_dir = "SLURM_outputs",
            resume_from_checkpoint=True
        ),
        train_dataset=data,
    )
    
    trainer.train()

    model.save_lora('SLURM_GRPO_lt')