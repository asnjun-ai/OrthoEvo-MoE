import torch
import torch.nn as nn
import torch.nn.functional as F
from router import ExplicitModalityAwareRouter

_CURRENT_MODALITY_MASK = None


def set_global_modality_mask(mask):
    global _CURRENT_MODALITY_MASK
    _CURRENT_MODALITY_MASK = mask


def clear_global_modality_mask():
    global _CURRENT_MODALITY_MASK
    _CURRENT_MODALITY_MASK = None


class Expert(nn.Module):
    """标准的 FFN/MLP 专家网络，兼容 LLaMA 的 w1/w2 命名与 LoRA"""

    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class SharedLinearRouter(nn.Module):
    """消融对比 1：传统单体共享线性路由（全模态 Token 共享同一个门控）"""

    def __init__(self, hidden_dim, total_experts, top_k=1):
        super().__init__()
        self.total_experts = total_experts
        self.top_k = top_k
        self.shared_gate = nn.Linear(hidden_dim, total_experts, bias=False)

    def forward(self, hidden_states, modality_mask=None):
        bsz, seq_len, _ = hidden_states.shape
        flat_x = hidden_states.view(-1, hidden_states.shape[-1])
        # 关键对齐：确保输入与门控层权重 dtype 一致
        flat_x = flat_x.to(dtype=self.shared_gate.weight.dtype)
        logits = self.shared_gate(flat_x)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        return weights.view(bsz, seq_len, self.top_k), indices.view(bsz, seq_len, self.top_k)


class DynamicTokenRouter(nn.Module):
    """消融对比 2：动态 Token 路由 (DTR，两层 MLP 软门控隐式混合)"""

    def __init__(self, hidden_dim, total_experts, top_k=1):
        super().__init__()
        self.total_experts = total_experts
        self.top_k = top_k
        self.dtr_proj = nn.Linear(hidden_dim, hidden_dim // 4, bias=False)
        self.dtr_act = nn.SiLU()
        self.dtr_gate = nn.Linear(hidden_dim // 4, total_experts, bias=False)

    def forward(self, hidden_states, modality_mask=None):
        bsz, seq_len, _ = hidden_states.shape
        flat_x = hidden_states.view(-1, hidden_states.shape[-1])
        # 关键对齐：确保输入与门控层权重 dtype 一致
        flat_x = flat_x.to(dtype=self.dtr_proj.weight.dtype)
        feat = self.dtr_act(self.dtr_proj(flat_x))
        logits = self.dtr_gate(feat)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        return weights.view(bsz, seq_len, self.top_k), indices.view(bsz, seq_len, self.top_k)


class OrthoEvoMoELayer(nn.Module):

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_vision_experts=2,
        num_text_experts=2,
        top_k=1,
        router_type="explicit",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_vision_experts = num_vision_experts
        self.num_text_experts = num_text_experts
        self.total_experts = num_vision_experts + num_text_experts
        self.top_k = top_k
        self.router_type = router_type

        # 1. 路由器模块架构分支
        if router_type == "shared_linear":
            self.router = SharedLinearRouter(
                hidden_dim, self.total_experts, top_k=top_k
            )
        elif router_type == "dtr":
            self.router = DynamicTokenRouter(
                hidden_dim, self.total_experts, top_k=top_k
            )
        else:  # explicit
            self.router = ExplicitModalityAwareRouter(
                hidden_dim=hidden_dim,
                num_vision_experts=num_vision_experts,
                num_text_experts=num_text_experts,
                top_k=top_k,
            )

        # 2. 统一专家列表 (0 ~ num_vision_experts-1 为视觉专家，之后为文本专家)
        self.experts = nn.ModuleList(
            [
                Expert(hidden_dim, intermediate_dim)
                for _ in range(self.total_experts)
            ]
        )

    # 关键兼容属性：消除 AttributeError
    @property
    def vision_experts(self):
        return self.experts[: self.num_vision_experts]

    @property
    def text_experts(self):
        return self.experts[self.num_vision_experts :]

    def get_expert_weights(self):
        """提取所有专家的第一层权重矩阵，用于计算正交惩罚损失"""
        return [expert.w1.weight for expert in self.experts]

    def forward(self, hidden_states, modality_mask=None):
        global _CURRENT_MODALITY_MASK
        
        orig_dtype = hidden_states.dtype  # 记录上游传进来的原始 dtype (如 float32)

        if modality_mask is None:
            modality_mask = _CURRENT_MODALITY_MASK

        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        if modality_mask is None:
            modality_mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)

        if modality_mask.shape != (batch_size, seq_len):
            if modality_mask.shape[0] == batch_size and modality_mask.shape[1] > seq_len:
                modality_mask = modality_mask[:, :seq_len]
            else:
                modality_mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)

        routing_weights, selected_experts = self.router(hidden_states, modality_mask)

        flat_inputs = hidden_states.view(-1, hidden_dim)
        flat_weights = routing_weights.view(-1, self.top_k)
        flat_experts = selected_experts.view(-1, self.top_k)

        # 保证与专家权重的精度一致 (float16)
        expert_dtype = self.experts[0].w1.weight.dtype
        flat_inputs_expert = flat_inputs.to(dtype=expert_dtype)
        flat_weights_expert = flat_weights.to(dtype=expert_dtype)

        final_output = torch.zeros_like(flat_inputs_expert)

        for i, expert in enumerate(self.experts):
            token_idx, topk_idx = torch.where(flat_experts == i)
            if token_idx.numel() == 0:
                continue

            expert_inputs = flat_inputs_expert[token_idx]
            weight = flat_weights_expert[token_idx, topk_idx].unsqueeze(-1)
            expert_output = expert(expert_inputs)
            final_output.index_add_(0, token_idx, expert_output * weight)

        # 还原回主干网络的原始 dtype 返回
        return final_output.view(batch_size, seq_len, hidden_dim).to(dtype=orig_dtype)