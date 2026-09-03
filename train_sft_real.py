import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像加速

import argparse
import gc
import torch
from torch.utils.data import DataLoader
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

from dataset import MLLMInstructionDataset, MLLMDataCollator
from loss import OrthogonalPenaltyLoss
from patch_model import convert_mllm_to_ortho_evomoe
from moe_layer import set_global_modality_mask, clear_global_modality_mask

def parse_args():
    parser = argparse.ArgumentParser(description="OrthoEvo-MoE SFT Training (8-Layer Safe Mode)")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--cache_dir", type=str, default="/mnt/z/work/SCI/cache")
    parser.add_argument("--json_path", type=str, default="./data/llava_instruct_exp1.json")
    parser.add_argument("--image_dir", type=str, default="./data/images")
    parser.add_argument("--output_dir", type=str, required=True, help="Checkpoint 保存路径")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_ortho", type=float, default=0.05, help="正交惩罚系数 λ")
    parser.add_argument("--sample_rows", type=int, default=256, help="正交采样行数")
    return parser.parse_args()

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"1. 正在加载底座模型: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map={"": 0},
        local_files_only=True
    )

    # 1. 挂载 4 层 MoE 专家层 (28~31 层，兼顾表征分化与 12GB 显存安全)
    target_layers = list(range(28, 32))
    model, moe_layers = convert_mllm_to_ortho_evomoe(
        model, 
        target_layer_indices=target_layers,
        num_vision_experts=2,
        num_text_experts=2,
        top_k=1
    )

    # 2. 启用 Gradient Checkpointing 核心显存压缩
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 3. 冻结底层基座并挂载 LoRA
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

    # 4. 构建数据流
    train_dataset = MLLMInstructionDataset(args.json_path, args.image_dir, processor)
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    collator = MLLMDataCollator(pad_token_id=pad_id)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True,
        collate_fn=collator
    )

    ortho_loss_fn = OrthogonalPenaltyLoss(sample_rows=args.sample_rows)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    print(f"\n🚀 开始训练: λ = {args.lambda_ortho} | 输出目录 = {args.output_dir} | 总批次数: {len(train_loader)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            labels = batch["labels"].to(device)
            modality_mask = batch["modality_mask"].to(device)

            set_global_modality_mask(modality_mask)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            l_lm = outputs.loss

            # 计算 MoE 层的正交正则损失和
            total_ortho_loss = 0.0
            if args.lambda_ortho > 0:
                for moe_layer in moe_layers:
                    expert_weights = moe_layer.get_expert_weights()
                    total_ortho_loss += ortho_loss_fn(expert_weights)

            l_total = l_lm + args.lambda_ortho * total_ortho_loss
            scaled_loss = l_total / args.grad_accum_steps
            scaled_loss.backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            clear_global_modality_mask()
            total_epoch_loss += l_total.item()

            if step % 10 == 0 or step == len(train_loader):
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}] | L_LM: {l_lm.item():.4f} | L_ortho: {float(total_ortho_loss):.4f} | L_total: {l_total.item():.4f}")

        avg_loss = total_epoch_loss / len(train_loader)
        print(f"🎉 Epoch [{epoch}/{args.epochs}] 完成! 平均 Loss: {avg_loss:.4f}")

        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        model.save_pretrained(epoch_dir)
        print(f"💾 Checkpoint 已保存至: {epoch_dir}\n")

if __name__ == "__main__":
    args = parse_args()
    train(args)