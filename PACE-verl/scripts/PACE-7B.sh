#!/bin/bash
set -euo pipefail

node_num=4
train_batch_size=64
save_freq=20
test_freq=20
total_epochs=10

model_path="/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

train_files='["../data/prefill_skywork-or1/prefill_skywork_or1.parquet"]'
test_files='["../data/aime/train.parquet"]'

max_prompt_length=$((1024 * 3))      
max_response_length=$((1024 * 16))   
ulysses_sequence_parallel_size=4
vllm_tp=4
ppo_mini_batch_size=64
gen_prompt_bsz=$(train_batch_size)  
gpu_memory_utilization=0.7
use_kl_loss=0
loss_agg_mode="seq-mean-token-mean"
format_required=1
rollout_num=8
rollout_backend="vllm"

save_path="YOUR SAVE PATH"
export TENSORBOARD_DIR="YOUR TENSORBOARD PATH"

reward_type="generative_length"
adv_estimator="grpo"
kl_loss_coef=0.01
offload=True

project_name="PACE"
exp_name="pace-1.5B"

temperature=1.0
top_p=1.0
top_k=-1   

use_dynamic_bsz=True

actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / ulysses_sequence_parallel_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / ulysses_sequence_parallel_size))


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=1 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.prefill_enable=True \
    data.gen_batch_size=${gen_prompt_bsz} \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${train_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${vllm_tp} \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.name=${rollout_backend} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    trainer.logger="['console','tensorboard']" \
    trainer.val_before_train=False \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${node_num} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${save_path}" \
    reward_model.reward_manager=${reward_type} \
    reward_model.reward_kwargs.format_required=${format_required} \
    trainer.resume_mode=auto \
    +trainer.reward_type=${reward_type}
