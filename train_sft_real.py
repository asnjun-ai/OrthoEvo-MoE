import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像加速

import gc
import torch
from torch.utils.data import DataLoader
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

from dataset import MLLMInstructionDataset
from loss import OrthogonalPenaltyLoss
from patch_model import convert_mllm_to_ortho_evomoe
from moe_layer import set_global_modality_mask, clear_global_modality_mask
from dataset import MLLMInstructionDataset, MLLMDataCollator

def train_sft_epoch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    # 路径参数 (替换为你的真实 JSON 和图片目录路径)
    json_path = "./data/llava_instruct_sample.json"  # 数据集 JSON 文件
    image_dir = "./data/images"                     # 图片跟目录
    output_dir = "./checkpoints/ortho_evomoe_lora" # 训练权重保存路径
    
    # 1. 超参数配置
    epochs = 3
    batch_size = 2
    learning_rate = 1e-4
    lambda_ortho = 0.05  # 正交惩罚系数 λ

    print("1. 正在加载基础模型与 Processor...")
    processor = AutoProcessor.from_pretrained(model_id, cache_dir="/mnt/z/work/SCI/cache")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir="/mnt/z/work/SCI/cache",
        torch_dtype=torch.float16,
        device_map={"": 0},
        local_files_only=True
    )

    # 2. 挂载 OrthoEvoMoELayer 专家层 (替换第 16~31 层)
    model, moe_layers = convert_mllm_to_ortho_evomoe(
        model, 
        target_layer_indices=list(range(16, 32)),
        num_vision_experts=2,
        num_text_experts=2,
        top_k=1
    )

    # 3. 准备 LoRA 模块
    for param in model.parameters():
        param.requires_grad = False
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["w1", "w2", "vision_gate", "text_gate"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

# 4. 加载真实 Dataset 与 DataLoader
    print("2. 正在加载真实数据集...")
    train_dataset = MLLMInstructionDataset(json_path, image_dir, processor)
    
    # 💡 传入 collate_fn 动态整理器，自动对齐不同长度样本
    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    collator = MLLMDataCollator(pad_token_id=pad_token_id)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        collate_fn=collator  # 👈 关键修复：加入动态填充整理器
    )

    # 5. 优化器与损失函数
    ortho_loss_fn = OrthogonalPenaltyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    print(f"\n🚀 开始真实 SFT 微调 (总 Epochs: {epochs}, Batches/Epoch: {len(train_loader)})...")

    # 6. Epoch 训练大循环
    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            labels = batch["labels"].to(device)
            modality_mask = batch["modality_mask"].to(device)

            # 设置全局上下文模态掩码
            set_global_modality_mask(modality_mask)
            optimizer.zero_grad()

            # MLLM 前向传播
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=labels
            )
            l_lm = outputs.loss

            # 计算挂载的各 MoE 层的正交损失 L_ortho
            total_ortho_loss = 0.0
            for moe_layer in moe_layers:
                expert_weights = moe_layer.get_expert_weights()
                total_ortho_loss += ortho_loss_fn(expert_weights)

            # 汇总总 Loss: L_total = L_LM + λ * ∑ L_ortho
            l_total = l_lm + lambda_ortho * total_ortho_loss

            # 反向传播与梯度更新
            l_total.backward()
            optimizer.step()
            clear_global_modality_mask()

            total_epoch_loss += l_total.item()

            if step % 5 == 0 or step == len(train_loader):
                print(f"Epoch [{epoch}/{epochs}] | Step [{step}/{len(train_loader)}] | L_LM: {l_lm.item():.4f} | L_ortho: {total_ortho_loss.item():.4f} | L_total: {l_total.item():.4f}")

        avg_loss = total_epoch_loss / len(train_loader)
        print(f"🎉 Epoch [{epoch}/{epochs}] 完成! 平均 Loss: {avg_loss:.4f}\n")

        # 7. 每个 Epoch 结束自动保存保存 Checkpoint 权重
        epoch_save_path = os.path.join(output_dir, f"epoch_{epoch}")
        model.save_pretrained(epoch_save_path)
        print(f"💾 Checkpoint 已保存至: {epoch_save_path}")

if __name__ == "__main__":
    train_sft_epoch()