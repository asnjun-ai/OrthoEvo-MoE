import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplicitModalityAwareRouter(nn.Module):
    """
    显式模态感知路由器：硬性分割视觉专家库 (E_V) 与 文本专家库 (E_T)
    杜绝 Modality Blending
    """
    def __init__(self, hidden_dim, num_vision_experts=2, num_text_experts=2, top_k=1):
        super().__init__()
        self.num_vision_experts = num_vision_experts
        self.num_text_experts = num_text_experts
        self.total_experts = num_vision_experts + num_text_experts
        self.top_k = top_k

        # 视觉与文本分别设立独立的轻量路由网络
        self.vision_gate = nn.Linear(hidden_dim, num_vision_experts, bias=False)
        self.text_gate = nn.Linear(hidden_dim, num_text_experts, bias=False)

    def forward(self, hidden_states, modality_mask):
        """
        hidden_states: [batch_size, seq_len, hidden_dim]
        modality_mask: [batch_size, seq_len], 1 表示 Vision Token, 0 表示 Text Token
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # 1. 核心修复：获取门控网络权重精度 (如 float16)，并将输入与填充张量统一对齐
        gate_dtype = self.vision_gate.weight.dtype
        fill_val = torch.finfo(gate_dtype).min if gate_dtype.is_floating_point else -1e4

        # 初始化 logits 矩阵，确保与门控权重 dtype 一致
        full_logits = torch.full(
            (batch_size, seq_len, self.total_experts),
            fill_value=fill_val,
            device=device,
            dtype=gate_dtype
        )

        # 2. 处理 Vision Tokens
        vis_indices = (modality_mask == 1)
        if vis_indices.any():
            # 转换至 gate_dtype 防止 float32 != float16 矩阵乘法报错
            vis_tokens = hidden_states[vis_indices].to(dtype=gate_dtype)
            vis_logits = self.vision_gate(vis_tokens)
            
            vis_slice = full_logits[:, :, :self.num_vision_experts].clone()
            vis_slice[vis_indices] = vis_logits
            full_logits[:, :, :self.num_vision_experts] = vis_slice

        # 3. 处理 Text Tokens
        text_indices = (modality_mask == 0)
        if text_indices.any():
            # 转换至 gate_dtype 防止 float32 != float16 矩阵乘法报错
            text_tokens = hidden_states[text_indices].to(dtype=gate_dtype)
            text_logits = self.text_gate(text_tokens)
            
            text_slice = full_logits[:, :, self.num_vision_experts:].clone()
            text_slice[text_indices] = text_logits
            full_logits[:, :, self.num_vision_experts:] = text_slice

        # 4. Softmax & Top-K 选择
        routing_weights = F.softmax(full_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # 权重归一化并转回与输入特征一致的精度
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        topk_weights = topk_weights.to(dtype=hidden_states.dtype)

        return topk_weights, topk_indices