import torch
import torch.nn as nn
import torch.optim as optim

from loss import OrthogonalPenaltyLoss
from moe_layer import OrthoEvoMoELayer

def train_convergence_demo():
    print("==================================================")
    print("🚀 开始 OrthoEvo-MoE 50-Step 收敛性模拟测试...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 运行设备: {device}")

    # 1. 超参数配置
    batch_size, seq_len, hidden_dim, intermediate_dim = 2, 8, 128, 512
    lambda_ortho = 0.05  # 正交惩罚系数 λ
    learning_rate = 1e-3
    total_steps = 50

    # 2. 实例化模型、Loss 与优化器
    moe_layer = OrthoEvoMoELayer(
        hidden_dim=hidden_dim, 
        intermediate_dim=intermediate_dim,
        num_vision_experts=2, 
        num_text_experts=2, 
        top_k=1
    ).to(device)

    ortho_loss_fn = OrthogonalPenaltyLoss()
    task_criterion = nn.MSELoss()
    optimizer = optim.AdamW(moe_layer.parameters(), lr=learning_rate)

    # 3. 构造固定的伪 Batch 数据 (用于测试单批次拟合)
    torch.manual_seed(42)  # 固定随机种子以便复现
    inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    targets = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    modality_mask = torch.tensor([[1,1,1,1, 0,0,0,0], 
                                  [1,1,1,1, 0,0,0,0]], device=device)

    print("\n--- 开始训练循环 ---")
    print(f"{'Step':^6} | {'L_task (主任务)':^15} | {'L_ortho (正交)':^15} | {'L_total (总损失)':^15}")
    print("-" * 60)

    initial_loss = None
    final_loss = None

    # 4. 50 Step 训练循环
    for step in range(1, total_steps + 1):
        optimizer.zero_grad()
        
        # 前向传播
        outputs = moe_layer(inputs, modality_mask)

        # Loss 计算
        l_task = task_criterion(outputs, targets)
        expert_weights = moe_layer.get_expert_weights()
        l_ortho = ortho_loss_fn(expert_weights)
        l_total = l_task + lambda_ortho * l_ortho

        # 记录初始与最终 Loss
        if step == 1:
            initial_loss = l_total.item()
        if step == total_steps:
            final_loss = l_total.item()

        # 反向传播与梯度更新
        l_total.backward()
        optimizer.step()

        # 每 5 个 Step 打印一次进度
        if step == 1 or step % 5 == 0:
            print(f"{step:^6d} | {l_task.item():^15.4f} | {l_ortho.item():^15.4f} | {l_total.item():^15.4f}")

    print("-" * 60)
    drop_rate = ((initial_loss - final_loss) / initial_loss) * 100
    print(f"🎉 训练完成！总 Loss 从 {initial_loss:.4f} 下降至 {final_loss:.4f} (下降了 {drop_rate:.2f}%)")
    print("==================================================")

if __name__ == "__main__":
    train_convergence_demo()