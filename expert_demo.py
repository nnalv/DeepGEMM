import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN
from functools import partial
from typing import Optional, Tuple
import deepgemm

fp8_cache={}

class DeepGEMM(Function):
    @staticmethod
    def forward(ctx, input, weight, input_fp8, input_scale, weight_fp8, weight_scale):
        print(f"forward called")
        ctx.save_for_backward(input, weight_fp8, weight_scale)
        output = deepgemm.gemm((input_fp8, input_scale), (weight_fp8, weight_scale))

        # ctx.save_for_backward(input, weight)
        # output = input @ weight.T
        # print(output)
        return output
    

    @staticmethod
    def backward(ctx, grad_output):
        print(f"backward called")
        input, weight_fp8, weight_scale = ctx.saved_tensors

        (grad_fp8, grad_scale) = deepgemm.per_token_cast_to_fp8(grad_output)
        grad_input = deepgemm.gemm((grad_fp8, grad_scale), (weight_fp8.T.contiguous(), weight_scale.T.contiguous()))

        grad_t_fp8, grad_t_scale = deepgemm.per_token_cast_to_fp8(grad_output.T.contiguous())
        input_t_fp8, input_t_scale = deepgemm.per_token_cast_to_fp8(input.T.contiguous())
        grad_weight = deepgemm.wgrad_gemm((grad_t_fp8, grad_t_scale), (input_t_fp8, input_t_scale))

        # input, weight = ctx.saved_tensors
        # grad_input = grad_output @ weight
        # grad_weight = grad_output.T @ input
        # return grad_input, grad_weight
        return grad_input, grad_weight, None ,None, None, None


class WeDeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_weight = nn.Parameter(torch.empty(self.intermediate_size, self.hidden_size, dtype=torch.bfloat16))
        self.up_weight = nn.Parameter(torch.empty(self.intermediate_size, self.hidden_size, dtype=torch.bfloat16))
        self.down_weight = nn.Parameter(torch.empty(self.hidden_size, self.intermediate_size, dtype=torch.bfloat16))
        
        self.act_fn = ACT2FN[config.hidden_act]
        
    def forward(self, x):
        (x_fp8, x_scale) = deepgemm.per_token_cast_to_fp8(x)

        self.gate_weight_fp8, self.gate_scale = deepgemm.per_block_cast_to_fp8(self.gate_weight.data)
        gate_output = DeepGEMM.apply(x, self.gate_weight, x_fp8, x_scale, self.gate_weight_fp8, self.gate_scale)
       
        self.up_weight_fp8, self.up_scale = deepgemm.per_block_cast_to_fp8(self.up_weight.data)
        up_output = DeepGEMM.apply(x, self.up_weight, x_fp8, x_scale, self.up_weight_fp8, self.up_scale)

        activated_output = self.act_fn(gate_output)
        activated_up_output = activated_output * up_output

        (activated_up_output_fp8, activated_up_output_scale) = deepgemm.per_token_cast_to_fp8(activated_up_output)
        self.down_weight_fp8, self.down_scale = deepgemm.per_block_cast_to_fp8(self.down_weight.data)

        down_proj = DeepGEMM.apply(activated_up_output, self.down_weight, activated_up_output_fp8, activated_up_output_scale, self.down_weight_fp8, self.down_scale)

        # gate_output = DeepGEMM.apply(x, self.gate_weight)
        # up_output = DeepGEMM.apply(x, self.up_weight)
        # activated_output = self.act_fn(gate_output)
        # activated_up_output = activated_output * up_output
        # down_proj = DeepGEMM.apply(activated_up_output, self.down_weight)
        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class DeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=torch.bfloat16)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # print(f"before gemm : x:{x}, w:{self.gate_proj.weight}")
        gate_output = self.gate_proj(x)
        # print(f"after gemm : x:{x}, w:{self.gate_proj.weight}")
        # if not self.config.is_print:
        #     print(f"DeepseekV3MLP-gate_output:{gate_output}")
        #     is_print=True
    
        down_proj = self.down_proj(self.act_fn(gate_output) * self.up_proj(x))
        return down_proj


class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts)))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config, we_mlp=False):
        super().__init__()
        self.config = config
        self.we_mlp = we_mlp
        self.experts = nn.ModuleList(
            [   
                WeDeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size) if we_mlp else DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size) 
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)
        

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                print(f"execute expert:{expert_idx}") 
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals) #residuals.shape: [128, 512, 2048]
        return hidden_states

class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # self.self_attn = ATTENTION_CLASSES[config._attn_implementation](
        #     config=config, layer_idx=layer_idx
        # )

        self.mlp = (
            DeepseekV3MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV3MLP(config)
        )

        # norm对于精度保持非常重要，避免OOD值的扰动
        self.input_layernorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_idx = layer_idx
        print(f"layer_idx: {layer_idx}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # hidden_states, self_attn_weights, present_key_value = self.self_attn(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_value=past_key_value,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        #     **kwargs,
        # )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


def compare_moe_layers(hf_model, we_model, rtol=1e-3, atol=1e-3):
    """Compare MoE layers between HuggingFace and Megatron models"""
    # Storage for intermediate activations
    hf_hiddens = [{} for _ in range(1)]  # Just one layer for simplicity
    we_hiddens = [{} for _ in range(1)]
    
    # Storage for gradients
    hf_grads = [{} for _ in range(1)]
    we_grads = [{} for _ in range(1)]
    
    # Define hooks for capturing inputs and outputs
    def print_input_hook(module, input, output=None, layer_idx=0, mode=None):
        if mode == "hf-experts_input":
            hf_hiddens[layer_idx]["experts_input"] = input[0]
        elif mode == "we-experts_input":
            we_hiddens[layer_idx]["experts_input"] = input[0]
        return None

    def print_output_hook(module, input, output, layer_idx=0, mode=None):
        if mode == "hf-experts_output":
            hf_hiddens[layer_idx]["experts_output"] = output
        elif mode == "we-experts_output":
            we_hiddens[layer_idx]["experts_output"] = output
        return None

    # Define backward hooks for capturing gradients
    def print_backward_hook(module, grad_input, grad_output, layer_idx=0, mode=None, name=None):
        if mode.startswith("hf-"):
            hf_grads[layer_idx][name] = grad_output[0].clone().detach()
        elif mode.startswith("we-"):
            we_grads[layer_idx][name] = grad_output[0].clone().detach()
        return None

    # Register hooks for HuggingFace model
    hf_output = None
    we_output = None
    
    if hf_model:
        # Assuming the MoE layer is in the first transformer layer
        hf_model.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=0, mode="hf-experts_input")
        )
        hf_model.register_forward_hook(
            partial(print_output_hook, layer_idx=0, mode="hf-experts_output")
        )
        # Register backward hooks
        hf_model.register_full_backward_hook(
            partial(print_backward_hook, layer_idx=0, mode="hf-experts", name="model_grad")
        )

    # Register hooks for Megatron model
    if we_model:
        we_model.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=0, mode="we-experts_input")
        )
        we_model.register_forward_hook(
            partial(print_output_hook, layer_idx=0, mode="we-experts_output")
        )
        # Register backward hooks
        we_model.register_full_backward_hook(
            partial(print_backward_hook, layer_idx=0, mode="we-experts", name="model_grad")
        )

    # Create test data
    batch_size = 128
    seq_len = 512
    hidden_size = hf_model.config.hidden_size if hf_model else we_model.config.hidden_size
    test_input = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True, dtype=torch.bfloat16)
    test_input_hf = test_input.clone().detach().requires_grad_(True)
    test_input_we = test_input.clone().detach().requires_grad_(True)
    
    # Register backward hooks on inputs
    def input_backward_hook(grad, layer_idx=0, mode=None):
        if mode == "hf":
            hf_grads[layer_idx]["input_grad"] = grad.clone().detach()
        elif mode == "we":
            we_grads[layer_idx]["input_grad"] = grad.clone().detach()
        return grad
    
    test_input_hf.register_hook(lambda grad: input_backward_hook(grad, layer_idx=0, mode="hf"))
    test_input_we.register_hook(lambda grad: input_backward_hook(grad, layer_idx=0, mode="we"))
    
    # Run forward and backward passes
    if hf_model:
        hf_model.cuda()
        hf_model.train()
        test_input_hf = test_input_hf.cuda()
        hf_output = hf_model(test_input_hf)
        
        # Register backward hooks on outputs
        def output_backward_hook(grad, layer_idx=0, mode=None):
            if mode == "hf":
                hf_grads[layer_idx]["output_grad"] = grad.clone().detach()
            elif mode == "we":
                we_grads[layer_idx]["output_grad"] = grad.clone().detach()
            return grad
        
        hf_output.register_hook(lambda grad: output_backward_hook(grad, layer_idx=0, mode="hf"))
        
        # Backward pass
        hf_loss = hf_output.sum()
        hf_loss.backward()
        hf_model.cpu()


    if we_model:
        we_model.cuda()
        we_model.train()
        test_input_we = test_input_we.cuda()
        # we_model_trace = torch.jit.trace(we_model, test_input_we)
        # print(we_model_trace.graph)

        we_output = we_model(test_input_we)
        
        # Register backward hooks on outputs
        we_output.register_hook(lambda grad: output_backward_hook(grad, layer_idx=0, mode="we"))        

        # Backward pass
        we_loss = we_output.sum()
        we_loss.backward()
        we_model.cpu()

    # Compare forward outputs
    epsilon = 1e-3
    for idx, (hfh, weh) in enumerate(zip(hf_hiddens, we_hiddens)):
        print(f"\n=== Layer {idx} Forward Results ===")
        for k, hfv in hfh.items():
            wev = weh[k]
            wev, hfv = wev.cpu(), hfv.cpu()
            same_num = (hfv != wev).sum()
            diff_num = ((hfv - wev).abs() > epsilon).sum()
            diff_max = (hfv - wev).abs().max()

            re = torch.allclose(wev.float().cpu(), hfv.float().cpu(), rtol=rtol, atol=atol)
            re_p = '\033[32mTrue\033[0m' if re else '\033[31mFalse\033[0m'
            
            diff_dg=calc_diff(wev, hfv)
            print(
                f"layer:{idx}, {k}, diff: {same_num}, diff>{epsilon}:[{diff_num}/{hfv.numel()}] "
                f"diff_max:{diff_max}, allclose: {re_p}, deep_gemm_diff:{diff_dg}"
            )

    # Compare final outputs
    if hf_output is not None and we_output is not None:
        print("\n=== Final Forward Output ===")
        hf_output = hf_output.cpu()
        we_output = we_output.cpu()
        same_num = (hf_output != we_output).sum()
        diff_num = ((hf_output - we_output).abs() > epsilon).sum()
        diff_max = (hf_output - we_output).abs().max()

        re = torch.allclose(we_output.float(), hf_output.float(), rtol=rtol, atol=atol)
        re_p = '\033[32mTrue\033[0m' if re else '\033[31mFalse\033[0m'

        diff_dg=calc_diff(wev, hfv)
        print(
            f"final output, diff: {same_num}, diff>{epsilon}:[{diff_num}/{hf_output.numel()}] "
            f"diff_max:{diff_max}, allclose: {re_p}, deep_gemm_diff:{diff_dg}"
        )
    
    # Compare backward gradients
    for idx, (hfg, weg) in enumerate(zip(hf_grads, we_grads)):
        print(f"\n=== Layer {idx} Backward Results ===")
        for k, hfv in hfg.items():
            if k in weg:
                wev = weg[k]
                wev, hfv = wev.cpu(), hfv.cpu()
                same_num = (hfv != wev).sum()
                diff_num = ((hfv - wev).abs() > epsilon).sum()
                diff_max = (hfv - wev).abs().max()

                re = torch.allclose(wev.float(), hfv.float(), rtol=rtol, atol=atol)
                re_p = '\033[32mTrue\033[0m' if re else '\033[31mFalse\033[0m'

                diff_dg=calc_diff(wev, hfv)
                print(
                    f"layer:{idx}, {k}, diff: {same_num}, diff>{epsilon}:[{diff_num}/{hfv.numel()}] "
                    f"diff_max:{diff_max}, allclose: {re_p}, deep_gemm_diff:{diff_dg}"
                )

    return hf_hiddens, we_hiddens

def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def compare_hf_mg_moe_models(hf_model_path, layer_idx=1):
    """
    比较HuggingFace和Megatron模型中的MoE层
    
    Args:
        hf_model_path: HuggingFace模型路径
        layer_idx: 要比较的层索引
    """
    # 加载HuggingFace模型配置
    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    setattr(hf_config, "is_print", False)
    
    # 创建我们的模型实例
    hf_layer = DeepseekV3MoE(hf_config)  # baseline
    we_layer = DeepseekV3MoE(hf_config, we_mlp=True)  # we model
    hf_layer.train()
    we_layer.train()
    
    for name, p in we_layer.named_parameters():
        print(f"we_layer: {name}, shape:{p.size()}")

    # for name, p in hf_layer.named_parameters():
    #     print(f"hf_layer: {name}, shape:{p.size()}")

    # 确保权重初始化相同
    for param_we, param_hf in zip(we_layer.parameters(), hf_layer.parameters()):
        if param_we is not None:
            param_we.data.copy_(param_hf.data)        

    # 比较两个模型
    hf_hiddens, we_hiddens = compare_moe_layers(hf_layer, we_layer)
    
    return hf_hiddens, we_hiddens


"""
执行: python 20250520_expert_demo.py
需要注意: baseline没有实现GroupGEMM，是以hf作为baseline进行精度对比，
在接入DeepGEMM的时候需要注意下DeepGEMM对于expert的权重布局的要求是什么
以transformengine的GGEMM的权重排布举例:
def convert_hf_model_to_te():
    for i, hfexpert in enumerate(hflayer.mlp.experts):
        linear_fc1_weighti = getattr(mglayer.mlp.experts.linear_fc1, 'weight' + str(i))
        gate_weight, up_weight = torch.split(linear_fc1_weighti, split_size_or_sections=args.moe_ffn_hidden_size)
        hfexpert.gate_proj.weight.copy_(gate_weight)
        hfexpert.up_proj.weight.copy_(up_weight)
        linear_fc2_weighti = getattr(mglayer.mlp.experts.linear_fc2, 'weight' + str(i))
        hfexpert.down_proj.weight.copy_(linear_fc2_weighti)
"""
if __name__ == "__main__":
    torch.cuda.empty_cache()
    local_path = "/data/shared/models/huggingface.co/moonshotai/Moonlight-16B-A3B/"
    hf_hiddens, we_hiddens = compare_hf_mg_moe_models(local_path, layer_idx=1)
