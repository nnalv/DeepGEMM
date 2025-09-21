import torch
import json
from collections import defaultdict
# from transformers import DeepseekV3ForCausalLM
from safetensors.torch import safe_open
import deepgemm

def load_moe_weight(hf_layer, we_model=False):
    with open("/data/users/nlv/DeepSeek-R1-5-layers/model.safetensors.index.json", "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]

    layer_idx = 3
    required_keys = [k for k in weight_map.keys() if k.startswith(f"model.layers.{layer_idx}.mlp.experts")]

    loaded_weights = {}
    for key in required_keys:
        file_name = weight_map[key]
        file_path = f"/data/users/nlv/DeepSeek-R1-5-layers/{file_name}"
    
        with safe_open(file_path, framework="pt", device="cpu") as f:
            weights = f.get_tensor(key)

            key_split = key.split(".")
            expert_idx = int(key_split[5])
            if expert_idx not in loaded_weights:
                loaded_weights[expert_idx] =  defaultdict(dict)

            loaded_weights[expert_idx][f"{key_split[6]}.{key_split[7]}"] = weights

    with torch.no_grad():
        for i, expert in enumerate(hf_layer.experts):
            gate_weight = loaded_weights[i]['gate_proj.weight']
            gate_scale = loaded_weights[i]['gate_proj.weight_scale_inv']
            gate_restored_bf16 = deepgemm.restore_per_block_fp8(gate_weight, gate_scale)
            if we_model:
                m, n = expert.gate_weight.data.shape
                expert.gate_weight.data.copy_(gate_restored_bf16[:m, :n])
            else:
                m, n = expert.gate_proj.weight.data.shape
                expert.gate_proj.weight.data.copy_(gate_restored_bf16[:m, :n])

            up_weight = loaded_weights[i]['up_proj.weight']
            up_scale = loaded_weights[i]['up_proj.weight_scale_inv']
            up_restored_bf16 = deepgemm.restore_per_block_fp8(up_weight, up_scale)
            if we_model:
                m, n = expert.up_weight.data.shape
                expert.up_weight.data.copy_(up_restored_bf16[:m, :n])
            else:
                m, n = expert.up_proj.weight.data.shape
                expert.up_proj.weight.data.copy_(up_restored_bf16[:m, :n])

            down_weight = loaded_weights[i]['down_proj.weight']
            down_scale = loaded_weights[i]['down_proj.weight_scale_inv']
            down_restored_bf16 = deepgemm.restore_per_block_fp8(down_weight, down_scale)
            if we_model:
                m, n = expert.down_weight.data.shape
                expert.down_weight.data.copy_(down_restored_bf16[:m, :n])
            else:
                m, n = expert.down_proj.weight.data.shape
                expert.down_proj.weight.data.copy_(down_restored_bf16[:m, :n])