import sys
sys.path.append("..")#

import re
import itertools
import math
import time
import random
from collections import Counter
from pprint import pprint
import matplotlib.pyplot as plt

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

    # this is already a PeftModelForCausalLM since that's
    # how it was saved in SFT_train - no need
    # to peftalize it
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "SLURM_finetuned_lt",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        # Enable vLLM fast inference
        fast_inference = True, 
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
                # In theory one could use vLLM for fast inference,
                # But at the time of writing this does not work:
                # https://github.com/unslothai/unsloth/issues/1877
                # use_vllm = True, 
                learning_rate=5e-6,
                adam_beta1=0.9,
                adam_beta2=0.99,
                weight_decay=0.1,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                optim="adamw_8bit",
                logging_steps=1,
                bf16=is_bfloat16_supported(),
                fp16=not is_bfloat16_supported(),
                per_device_train_batch_size=1,
                # Increase to 4 for smoother training
                gradient_accumulation_steps=1, 
                # Decrease if out of memory
                num_generations=32, 
                max_prompt_length=256,
                max_completion_length=32,
                # Set to 1 for a full training run
                num_train_epochs=1,
                max_steps=5000,
                save_steps=100,
                max_grad_norm=0.1,
                # Can use Weights & Biases
                report_to="none", 
                output_dir="SLURM_outputs",
                resume_from_checkpoint=True
            ),
        train_dataset=data,
    )
    
    trainer.train()
    trainer.save_model('SLURM_GRPO')
    trainer.save_state()

    df_history = pd.DataFrame(trainer.state.log_history)
    smoothed_rewards = df_history['rewards/direct_lt_correctness_reward_func'].rolling(window=50).mean()

    # Plotting the raw reward and the trend (smoothed reward)
    plt.figure(figsize=(20, 5))
    plt.plot(df_history.index, smoothed_rewards, label="Trend (Moving Average)", color="red", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.ylim(0,3)
    plt.title("Reward Trend using Moving Average")
    plt.legend()
    plt.savefig('./progress.png', dpi=300)
