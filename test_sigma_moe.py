import math
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import numpy as np
from model_sigma_tiktoken.modeling_sigma import ACT2FN, SigmaMLP
import torch.profiler as profiler

class FusedSigmaMLP(nn.Module):
    def __init__(self, config, n_routed_experts, hidden_size = None, intermediate_size = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.n_routed_experts = n_routed_experts
        self.gate_proj = nn.Parameter(torch.empty(self.n_routed_experts, self.hidden_size, self.intermediate_size))
        self.up_proj = nn.Parameter(torch.empty(self.n_routed_experts, self.hidden_size, self.intermediate_size))
        self.down_proj = nn.Parameter(torch.empty(self.n_routed_experts, self.intermediate_size, self.hidden_size))
        self.act_fn = ACT2FN[config.hidden_act]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init

        for i in range(self.n_routed_experts):
            init.kaiming_uniform_(self.gate_proj[i], a=math.sqrt(5))
            init.kaiming_uniform_(self.up_proj[i], a=math.sqrt(5))
            init.kaiming_uniform_(self.down_proj[i], a=math.sqrt(5))

    def forward(self, x):
        down_proj = torch.matmul(self.act_fn(torch.matmul(x, self.gate_proj)) * torch.matmul(x, self.up_proj), self.down_proj)
        return down_proj


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = 1
        self.topk_group = 1
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts)))
    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))


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


class SigmaMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        SigmaMLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList(
                [
                    SigmaMLP(
                        config, intermediate_size=config.moe_intermediate_size
                    )
                    for i in range(config.n_routed_experts)
                ]
            )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = SigmaMLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        identity = hidden_states 
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1]) 
        flat_topk_idx = topk_idx.view(-1) 
        if self.training:
            hidden_states = hidden_states.repeat_interleave(   
                self.num_experts_per_tok, dim=0
            )
            print(f"hidden_states: {hidden_states.shape}")
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            # y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


class SigmaMoE_Acce(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        SigmaMLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = FusedSigmaMLP(
                config, n_routed_experts=config.n_routed_experts
            )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = SigmaMLP(
                config=config, intermediate_size=intermediate_size
            )


    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.zeros_like(hidden_states)
            sorted_flat_topk_idx, sorted_idx = flat_topk_idx.sort()
            sorted_hidden_states = hidden_states.index_select(0, sorted_idx)
            activated_expert_idx, counts = sorted_flat_topk_idx.unique_consecutive(return_counts=True)
            per_expert_hidden_states = torch.split(sorted_hidden_states, counts.tolist())
            max_len = counts.max().item()
            
            if activated_expert_idx.shape[0] != self.experts_per_rank:
                padded_per_expert_hidden_states = []
                cur_expert_idx = 0
                for _activated_expert_idx, _expert_hidden_states in zip(activated_expert_idx, per_expert_hidden_states):
                    while _activated_expert_idx != cur_expert_idx:
                        padded_per_expert_hidden_states.append(torch.zeros((max_len, _expert_hidden_states.shape[-1]),
                                                                           dtype=_expert_hidden_states.dtype,
                                                                           device=_expert_hidden_states.device))
                        cur_expert_idx += 1
                    padded_per_expert_hidden_states.append(F.pad(_expert_hidden_states, (0, 0, 0, max_len - _expert_hidden_states.shape[0])))
                    cur_expert_idx += 1
                for _ in range(cur_expert_idx, self.experts_per_rank):
                    padded_per_expert_hidden_states.append(torch.zeros((max_len, _expert_hidden_states.shape[-1]),
                                                                       dtype=_expert_hidden_states.dtype,
                                                                       device=_expert_hidden_states.device))
                padded_per_expert_hidden_states = torch.stack(padded_per_expert_hidden_states)
            else:
                padded_per_expert_hidden_states = torch.stack([F.pad(t, (0, 0, 0, max_len - t.shape[0])) for t in per_expert_hidden_states], dim=0)

            per_expert_output = self.experts(padded_per_expert_hidden_states)
            valid_per_expert_output = torch.cat([
                per_expert_output[_expert_idx, :_count] for _expert_idx, _count in zip(activated_expert_idx, counts)
            ], dim=0)
            y.scatter_add_(0, sorted_idx.unsqueeze(-1).expand_as(sorted_hidden_states), valid_per_expert_output)

            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.type(hidden_states.dtype)
            y =  y.view(*orig_shape)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


if __name__ == "__main__":
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("model_sigma_tiktoken", trust_remote_code=True)
    # initialize the moe module
    moe = SigmaMoE(config).to("cuda:0")
    moe_acce = SigmaMoE_Acce(config).to("cuda:1")

    hidden_states = torch.randn(2, 4096, 2048).to("cuda:0")
    hidden_states_acce = hidden_states.to("cuda:1")

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        output = moe(hidden_states)
        output_2 = moe_acce(hidden_states_acce)

        print(output)
        print(output_2)

    # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))