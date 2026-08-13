import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像加速

import gc
import torch
import torch.nn as nn
from transformers import LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model

from loss import OrthogonalPenaltyLoss
from patch_model import convert_mllm_to_ortho_evomoe
from modality_utils import get_modality_mask
from moe_layer import set_global_modality_mask, clear_global_modality_mask

def train_lora_mllm():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "llava-hf/llava-1.5-7b-hf"

    print("1. 正在以 FP16 半精度加载 7B 基础模型...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        cache_dir="/mnt/z/work/SCI/cache",
        torch_dtype=torch.float16,   # 强制 FP16 加载，显存直接砍半 (从 28G 降至 14G)
        device_map={"": 0},
        low_cpu_mem_usage=True       # 开启 CPU 内存低消耗模式
    )

    # 2. 挂载 OrthoEvoMoELayer (替换 16~31 层)
    model, moe_layers = convert_mllm_to_ortho_evomoe(
        model, 
        target_layer_indices=list(range(16, 32)),
        num_vision_experts=2,
        num_text_experts=2,
        top_k=1
    )

    # 💡 关键内存优化：在注入 LoRA 前先手动冻结主干梯度并清理垃圾回收
    print("🧹 正在清理显存与内存碎片...")
    for param in model.parameters():
        param.requires_grad = False
    
    # 开启梯度检查点 (Gradient Checkpointing) 节省 60% 以上显存
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    gc.collect()
    torch.cuda.empty_cache()

    # 3. 注入 LoRA 模块
    print("2. 正在注入轻量化 LoRA 模块...")
    peft_config = LoraConfig(
        r=8,                       # 稍微调小 rank (从 16 降到 8)，足以表达专家特征且极其省显存
        lora_alpha=16,
        target_modules=["w1", "w2", "vision_gate", "text_gate"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # 确保准备训练的层被开启梯度
    model.print_trainable_parameters()

    # 4. 优化器与 Loss 准备
    ortho_loss_fn = OrthogonalPenaltyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    lambda_ortho = 0.05

# 5. 模拟单步前向传播
    print("\n3. 开始前向与反向传播迭代测试...")
    dummy_input_ids = torch.randint(0, 32000, (1, 64), device=device)
    dummy_input_ids[0, 10:30] = 32000  # 假设中间为 Image Token

    modality_mask = get_modality_mask(dummy_input_ids, image_token_id=32000)

    # 💡 设置全局模态掩码，供底层挂载的 OrthoEvoMoELayer 自动读取
    set_global_modality_mask(modality_mask)

    optimizer.zero_grad()

    outputs = model(input_ids=dummy_input_ids, labels=dummy_input_ids)
    l_lm = outputs.loss

    total_ortho_loss = 0.0
    for moe_layer in moe_layers:
        expert_weights = moe_layer.get_expert_weights()
        total_ortho_loss += ortho_loss_fn(expert_weights)

    l_total = l_lm + lambda_ortho * total_ortho_loss

    l_total.backward()
    optimizer.step()

    # 运行完毕清理全局变量
    clear_global_modality_mask()

    print(f"🎉 7B 真实多模态大模型挂载 + LoRA 微调成功!")
    print(f"📊 L_LM (语言损失): {l_lm.item():.4f} | L_ortho (正交损失和): {total_ortho_loss.item():.4f} | Total Loss: {l_total.item():.4f}")
    
if __name__ == "__main__":
    train_lora_mllm()