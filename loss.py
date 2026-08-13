import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalPenaltyLoss(nn.Module):
    """
    内存优化版正交惩罚损失函数：
    针对 12GB 消费级显卡，采用随机行采样（Row Subsampling）策略，
    将 11008x11008 大矩阵乘法降维，彻底防止显存爆炸（OOM）。
    """
    def __init__(self, sample_rows=256):
        super().__init__()
        self.sample_rows = sample_rows  # 限制参与计算的最大行数

    def forward(self, expert_weights):
        if not expert_weights or len(expert_weights) < 2:
            return torch.tensor(0.0, device=expert_weights[0].device if expert_weights else 'cpu')

        num_experts = len(expert_weights)
        ortho_loss = 0.0
        pair_count = 0

        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                w1 = expert_weights[i].float()
                w2 = expert_weights[j].float()

                w1_norm = F.normalize(w1, p=2, dim=-1)
                w2_norm = F.normalize(w2, p=2, dim=-1)

                # 💡 显存优化：随机采样部分行计算正交性，将显存占用降低 90% 以上
                if w1_norm.shape[0] > self.sample_rows:
                    indices = torch.randperm(w1_norm.shape[0], device=w1_norm.device)[:self.sample_rows]
                    w1_sub = w1_norm[indices]
                    w2_sub = w2_norm[indices]
                else:
                    w1_sub = w1_norm
                    w2_sub = w2_norm

                cos_sim_matrix = torch.matmul(w1_sub, w2_sub.T)
                pair_loss = torch.mean(cos_sim_matrix ** 2)

                ortho_loss += pair_loss
                pair_count += 1

        if pair_count > 0:
            ortho_loss = ortho_loss / pair_count

        return ortho_loss