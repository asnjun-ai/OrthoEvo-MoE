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

        # 💡 安全获取当前 FP16/BF16/FP32 的数值极小值，防止 -1e9 在 FP16 爆溢出
        fill_val = torch.finfo(hidden_states.dtype).min if hidden_states.dtype.is_floating_point else -1e4

        # 初始化 logits 矩阵
        full_logits = torch.full(
            (batch_size, seq_len, self.total_experts),
            fill_value=fill_val,
            device=device,
            dtype=hidden_states.dtype
        )

        # 1. 处理 Vision Tokens
        vis_indices = (modality_mask == 1)
        if vis_indices.any():
            vis_tokens = hidden_states[vis_indices]
            vis_logits = self.vision_gate(vis_tokens)
            
            vis_slice = full_logits[:, :, :self.num_vision_experts]
            vis_slice[vis_indices] = vis_logits
            full_logits[:, :, :self.num_vision_experts] = vis_slice

        # 2. 处理 Text Tokens
        text_indices = (modality_mask == 0)
        if text_indices.any():
            text_tokens = hidden_states[text_indices]
            text_logits = self.text_gate(text_tokens)
            
            text_slice = full_logits[:, :, self.num_vision_experts:]
            text_slice[text_indices] = text_logits
            full_logits[:, :, self.num_vision_experts:] = text_slice

        # 3. Softmax & Top-K 选择
        routing_weights = F.softmax(full_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        return topk_weights, topk_indices