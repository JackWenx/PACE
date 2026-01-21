from verl import DataProto
from typing import List, Dict, Tuple
import torch
import numpy as np


def add_prefill(batch: DataProto, tokenizer, config=None, idx2prefill={}, global_steps=500) -> DataProto:
    tensor_batch = batch.batch
    input_key = 'input_ids' if 'input_ids' in tensor_batch else 'prompts'
    origin_input_ids = tensor_batch[input_key]
    origin_attention_mask = tensor_batch['attention_mask']
    device = origin_input_ids.device
    batch_size = origin_input_ids.shape[0]
    
    extra_infos = batch.non_tensor_batch.get('extra_info', None)
    
    processed_samples = []
    new_raw_prompt_ids = []
    prefill_id_length = []

    for i in range(batch_size):
        curr_input_ids = origin_input_ids[i]
        curr_mask = origin_attention_mask[i]

        valid_indices = torch.nonzero(curr_mask).squeeze()
        if valid_indices.numel() == 0:
             valid_input_ids = torch.tensor([], device=device, dtype=torch.long)
        elif valid_indices.ndim == 0:
             valid_input_ids = curr_input_ids[valid_indices].unsqueeze(0)
        else:
             valid_input_ids = curr_input_ids[valid_indices]

        prefix_ids = []
        if extra_infos is not None:
            prefill_length = idx2prefill.get(extra_infos[i].get('index', -1), 512)
            prefill_text = extra_infos[i].get("prefill_text", "")
            
            left_forward = global_steps // 20
            prefill_length = prefill_length / pow(2, left_forward)
            # Debug info 
            print(f'prefill_length: {prefill_length}')
            
            if prefill_text:
                cur_length = 0
                if prefill_length > 32:                    
                    for chunk in prefill_text.split('\n\n'):
                        chunk_ids = tokenizer.encode(chunk + '\n\n', add_special_tokens=False)
                        if cur_length + len(chunk_ids) <= prefill_length:
                            prefix_ids.extend(chunk_ids)
                            cur_length += len(chunk_ids)
                        else:
                            break
            prefill_id_length.append(cur_length)
        
        if prefix_ids:
            prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=device)
            prefix_full_tensor = torch.cat([valid_input_ids, prefix_tensor], dim=0)
            
            len_prompt = len(prefix_full_tensor)
            if config:
                max_prompt_length = config.data.max_prompt_length
                if len_prompt > max_prompt_length:
                    prefix_full_tensor = prefix_full_tensor[:max_prompt_length]
        else:
            prefix_full_tensor = valid_input_ids
            
        processed_samples.append(prefix_full_tensor)
        new_raw_prompt_ids.append(prefix_full_tensor.cpu().tolist())

    if config is None:
        raise ValueError("Config must be provided to determine max_prompt_length")
        
    max_len = config.data.max_prompt_length
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    final_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    final_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for i, sample in enumerate(processed_samples):
        seq_len = sample.shape[0]
        
        if seq_len > max_len:
            sample = sample[:max_len]
            seq_len = max_len
            
        # calculate left padding start_idx
        start_idx = max_len - seq_len
        
        final_input_ids[i, start_idx:] = sample
        final_attention_mask[i, start_idx:] = 1

    # ------------------ Debug ------------------
    # print(f"\n{'='*20} Debug: Decoded Final Input IDs {'='*20}")
    # try:
    #     decoded_texts = tokenizer.batch_decode(final_input_ids, skip_special_tokens=True)
    #     target_eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    #     for idx, text in enumerate(decoded_texts):
    #         sample_ids = final_input_ids[idx]
    #         total_len = sample_ids.numel()
    #         valid_len = final_attention_mask[idx].sum().item()
            
    #         print(f"--- Sample {idx} | Total Len: {total_len} | Valid Len (Mask==1): {valid_len} ---")
    #         print(f"--- Sample {idx} ---")
    #         print(f'[original]: {tokenizer.decode(origin_input_ids[idx], skip_special_tokens=True)}')
    #         print(f'[cur]: {text}')
    #         print("-" * 20)
    # except Exception as e:
    #     print(f"Decode failed: {e}")
    # print(f"{'='*60}\n")
    # ------------------------------------------------

    # Position IDs
    position_ids = torch.cumsum(final_attention_mask, dim=1) - 1
    position_ids.masked_fill_(final_attention_mask == 0, 0)
    
    batch.batch[input_key] = final_input_ids
    if input_key == 'prompts' and 'input_ids' in batch.batch:
         batch.batch['input_ids'] = final_input_ids

    batch.batch['attention_mask'] = final_attention_mask
    
    if 'position_ids' in batch.batch:
        batch.batch['position_ids'] = position_ids
    
    if 'raw_prompt_ids' in batch.non_tensor_batch:
        batch.non_tensor_batch['raw_prompt_ids'] = np.array(new_raw_prompt_ids, dtype=object)

    # print(f'prefill_id_lenth: {len(prefill_id_length)}')
    # print(f'prefill: {prefill_id_length}')
    batch.non_tensor_batch["prefill_lengths"] = np.array(prefill_id_length, dtype=object)
    batch.non_tensor_batch["global_steps"] = np.array([global_steps] * batch_size, dtype=object)
    print(f'add_prefill done with manual left-padding!')
    return batch



def post_process_prefill(batch: DataProto, tokenizer, config) -> DataProto:
    tensor_batch = batch.batch
    non_tensor_batch = batch.non_tensor_batch

    prompts_ids = tensor_batch["prompts"]     
    responses = tensor_batch["responses"]      
    

    prefill_lengths = non_tensor_batch.get("prefill_lengths", [0] * prompts_ids.shape[0])

    if len(prefill_lengths) == 0:
        print("Warning: No prefill lengths found, skipping post_process.")
        return batch

    device = prompts_ids.device
    batch_size = prompts_ids.shape[0]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    new_prompt_list = []
    new_response_list = []

    for i in range(batch_size):
        curr_prompt = prompts_ids[i]
        valid_p_indices = torch.nonzero(curr_prompt != pad_token_id).squeeze()
        if valid_p_indices.ndim == 0 and valid_p_indices.numel() > 0:
             valid_p_indices = valid_p_indices.unsqueeze(0)
             
        if valid_p_indices.numel() > 0:
            valid_prompt = curr_prompt[valid_p_indices] # [Original + Prefill]
        else:
            valid_prompt = torch.tensor([], device=device, dtype=torch.long)

        
        p_len = int(prefill_lengths[i])
        
        
        if p_len > 0 and len(valid_prompt) >= p_len:
            original_prompt_ids = valid_prompt[:-p_len] 
            prefill_ids = valid_prompt[-p_len:]
        else:
            original_prompt_ids = valid_prompt
            prefill_ids = torch.tensor([], device=device, dtype=torch.long)

        curr_response = responses[i]
        valid_r_indices = torch.nonzero(curr_response != pad_token_id).squeeze()
        if valid_r_indices.ndim == 0 and valid_r_indices.numel() > 0:
             valid_r_indices = valid_r_indices.unsqueeze(0)

        if valid_r_indices.numel() > 0:
            valid_response = curr_response[valid_r_indices]
        else:
            valid_response = torch.tensor([], device=device, dtype=torch.long)

        new_response = torch.cat([prefill_ids, valid_response], dim=0)

        new_prompt_list.append(original_prompt_ids)
        new_response_list.append(new_response)

    max_prompt_len = config.data.max_prompt_length
    max_response_len = config.data.max_response_length

    new_prompts_tensor = torch.full((batch_size, max_prompt_len), pad_token_id, dtype=torch.long, device=device)
    new_responses_tensor = torch.full((batch_size, max_response_len), pad_token_id, dtype=torch.long, device=device)
    
    new_prompts_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.long, device=device)
    new_responses_mask = torch.zeros((batch_size, max_response_len), dtype=torch.long, device=device)

    for i in range(batch_size):
        p_seq = new_prompt_list[i]
        p_len = len(p_seq)
        if p_len > max_prompt_len:
            p_seq = p_seq[:max_prompt_len] # truncate
            p_len = max_prompt_len
        
        if p_len > 0:
            start_idx = max_prompt_len - p_len
            new_prompts_tensor[i, start_idx:] = p_seq
            new_prompts_mask[i, start_idx:] = 1

        # Response (Right Padding)
        r_seq = new_response_list[i]
        r_len = len(r_seq)
        if r_len > max_response_len:
            r_seq = r_seq[:max_response_len] # truncate
            r_len = max_response_len
        
        if r_len > 0:
            new_responses_tensor[i, :r_len] = r_seq
            new_responses_mask[i, :r_len] = 1


    final_input_ids = torch.cat([new_prompts_tensor, new_responses_tensor], dim=1)
    final_attention_mask = torch.cat([new_prompts_mask, new_responses_mask], dim=1)


    position_ids = torch.cumsum(final_attention_mask, dim=1) - 1
    position_ids.masked_fill_(final_attention_mask == 0, 0)

    tensor_batch["prompts"] = new_prompts_tensor
    tensor_batch["responses"] = new_responses_tensor
    tensor_batch["input_ids"] = final_input_ids
    tensor_batch["attention_mask"] = final_attention_mask
    
    if "position_ids" in tensor_batch:
        tensor_batch["position_ids"] = position_ids


    if 'raw_prompt_ids' in non_tensor_batch:
         updated_raw_prompts = []
         for p_tensor in new_prompt_list:
             updated_raw_prompts.append(p_tensor.cpu().tolist())
         non_tensor_batch['raw_prompt_ids'] = np.array(updated_raw_prompts, dtype=object)

    print("post_process_prefill done: Prefill moved from prompts to responses.")
    return batch




