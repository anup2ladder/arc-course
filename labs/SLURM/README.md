# Server stuff

This folder contains some code for LLM simulation on a SLURM server. Unfortunately my laptop is not big enough for some of the things I want to do!

Run first job_SFT.sh, which does supervised fine tuning and creates a folder "SLURM_finetuned_lt" with the LORA weights.

Then, run job_GRPO.sh, which does the RL training to learn how to actually solve the task.