import torch
import torch.nn as nn
import torch.nn.functional as F
from router import ExplicitModalityAwareRouter

# 全局上下文管理器，用于在 MLLM 前向传播时传递 modality_mask
_CURRENT_MODALITY_MASK = None

def set_global_modality_mask(mask):
    global _CURRENT_MODALITY_MASK
    _CURRENT_MODALITY_MASK = mask

def clear_global_modality_mask():
    global _CURRENT_MODALITY_MASK
    _CURRENT_MODALITY_MASK = None


class Expert(nn.Module):
    """标准的 FFN/MLP 专家网络"""
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        # 兼容 LLaMA 架构的命名规范
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()  # LLaMA 默认使用 SiLU (SwiGLU 核心)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class OrthoEvoMoELayer(nn.Module):
    """
    OrthoEvo-MoE 核心层：兼容 HuggingFace 单参数调用规范
    """
    def __init__(self, hidden_dim, intermediate_dim, num_vision_experts=2, num_text_experts=2, top_k=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_vision_experts = num_vision_experts
        self.num_text_experts = num_text_experts
        self.total_experts = num_vision_experts + num_text_experts
        self.top_k = top_k

        # 1. 实例化显式路由器
        self.router = ExplicitModalityAwareRouter(
            hidden_dim=hidden_dim,
            num_vision_experts=num_vision_experts,
            num_text_experts=num_text_experts,
            top_k=top_k
        )

        # 2. 构建专家库
        self.experts = nn.ModuleList([
            Expert(hidden_dim, intermediate_dim) for _ in range(self.total_experts)
        ])

    def get_expert_weights(self):
        """提取所有专家的第一层权重矩阵，用于计算正交惩罚损失"""
        return [expert.w1.weight for expert in self.experts]

    def forward(self, hidden_states, modality_mask=None):
        """
        兼容 HuggingFace self.mlp(hidden_states) 的单参数调用
        """
        global _CURRENT_MODALITY_MASK
        
        # 如果未直接传入 modality_mask，则优先获取全局注册的 mask
        if modality_mask is None:
            modality_mask = _CURRENT_MODALITY_MASK

        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # 如果依然没有提供 mask，默认降级为全部视为 Text Token (全 0)
        if modality_mask is None:
            modality_mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)

        # 校验维度匹配，若不适配则进行截断或自动补全
        if modality_mask.shape != (batch_size, seq_len):
            if modality_mask.shape[0] == batch_size and modality_mask.shape[1] > seq_len:
                modality_mask = modality_mask[:, :seq_len]
            else:
                modality_mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)

        # 获取 Top-K 路由权重与选中的专家索引
        routing_weights, selected_experts = self.router(hidden_states, modality_mask)

        # 展平 Tensor 方便批量分发计算
        flat_inputs = hidden_states.view(-1, hidden_dim)
        flat_weights = routing_weights.view(-1, self.top_k)
        flat_experts = selected_experts.view(-1, self.top_k)

        final_output = torch.zeros_like(flat_inputs)

        # 遍历每一个专家，稀疏计算
        for i, expert in enumerate(self.experts):
            token_idx, topk_idx = torch.where(flat_experts == i)
            if token_idx.numel() == 0:
                continue

            expert_inputs = flat_inputs[token_idx]
            weight = flat_weights[token_idx, topk_idx].unsqueeze(-1)

            expert_output = expert(expert_inputs)
            final_output.index_add_(0, token_idx, expert_output * weight)

        return final_output.view(batch_size, seq_len, hidden_dim)