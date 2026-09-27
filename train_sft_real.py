import argparse
import gc
import os
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

from dataset import MLLMDataCollator, MLLMInstructionDataset
from loss import OrthogonalPenaltyLoss
from moe_layer import clear_global_modality_mask, set_global_modality_mask
from patch_model import convert_mllm_to_ortho_evomoe

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像加速


def parse_args():
  parser = argparse.ArgumentParser(
      description=(
          "OrthoEvo-MoE SFT Training (Exp 5 Momentum Beta Supported)"
      )
  )
  parser.add_argument(
      "--model_id", type=str, default="llava-hf/llava-1.5-7b-hf"
  )
  parser.add_argument("--cache_dir", type=str, default="/mnt/z/work/SCI/cache")
  parser.add_argument(
      "--json_path", type=str, default="./data/llava_instruct_exp1.json"
  )
  parser.add_argument("--image_dir", type=str, default="./data/images")
  parser.add_argument(
      "--output_dir", type=str, required=True, help="Checkpoint 保存路径"
  )
  parser.add_argument("--epochs", type=int, default=3)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--grad_accum_steps", type=int, default=8)
  parser.add_argument("--lr", type=float, default=2e-4)
  parser.add_argument(
      "--lambda_ortho", type=float, default=0.05, help="正交惩罚系数 λ"
  )
  parser.add_argument(
      "--sample_rows", type=int, default=256, help="正交采样行数"
  )
  parser.add_argument(
      "--router_type",
      type=str,
      default="explicit",
      choices=["shared_linear", "dtr", "explicit"],
      help="路由器架构: shared_linear / dtr / explicit",
  )
  # 实验 5：专家演化动量系数 beta (0.0 表示不使用动量平滑)
  parser.add_argument(
      "--momentum_beta",
      type=float,
      default=0.80,
      help="专家权重演化动量系数 beta (例如: 0.95, 0.80, 0.60)",
  )
  return parser.parse_args()


def train(args):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  os.makedirs(args.output_dir, exist_ok=True)

  print(f"1. 正在加载分词与图像处理器: {args.model_id}")
  processor = AutoProcessor.from_pretrained(
      args.model_id, cache_dir=args.cache_dir, local_files_only=True
  )
  gc.collect()
  torch.cuda.empty_cache()

  # 配置 4-bit 量化底座
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True,
  )

  print(f"2. 正在加载 4-bit 量化底座模型...")
  model = LlavaForConditionalGeneration.from_pretrained(
      args.model_id,
      cache_dir=args.cache_dir,
      quantization_config=bnb_config,
      device_map={"": 0},
      low_cpu_mem_usage=True,
      local_files_only=True,
  )

  model = prepare_model_for_kbit_training(model)

  # 挂载 4 层 MoE 专家层
  target_layers = list(range(28, 32))
  model, moe_layers = convert_mllm_to_ortho_evomoe(
      model,
      target_layer_indices=target_layers,
      num_vision_experts=2,
      num_text_experts=2,
      top_k=1,
      router_type=args.router_type,
  )

  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()

  for param in model.parameters():
    param.requires_grad = False

  target_modules = ["w1", "w2"]
  if args.router_type == "shared_linear":
    target_modules += ["shared_gate"]
  elif args.router_type == "dtr":
    target_modules += ["dtr_proj", "dtr_gate"]
  else:  # explicit
    target_modules += ["vision_gate", "text_gate"]

  peft_config = LoraConfig(
      r=8,
      lora_alpha=16,
      target_modules=target_modules,
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
  )
  model = get_peft_model(model, peft_config)
  model.print_trainable_parameters()

  # 构建数据流
  train_dataset = MLLMInstructionDataset(
      args.json_path, args.image_dir, processor
  )
  pad_id = (
      processor.tokenizer.pad_token_id
      if processor.tokenizer.pad_token_id is not None
      else 0
  )
  collator = MLLMDataCollator(pad_token_id=pad_id)
  train_loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      drop_last=True,
      num_workers=0,
      pin_memory=False,
      collate_fn=collator,
  )

  ortho_loss_fn = OrthogonalPenaltyLoss(sample_rows=args.sample_rows)
  optimizer = torch.optim.AdamW(
      filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
  )

  print(
      f"\n🚀 开始训练: Router = {args.router_type} | λ = {args.lambda_ortho} | "
      f"Momentum β = {args.momentum_beta} | 批次数: {len(train_loader)}"
  )

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

      outputs = model(
          input_ids=input_ids, pixel_values=pixel_values, labels=labels
      )
      l_lm = outputs.loss

      # 正交正则化损失
      total_ortho_loss = 0.0
      if args.lambda_ortho > 0:
        for moe_layer in moe_layers:
          expert_weights = moe_layer.get_expert_weights()
          total_ortho_loss += ortho_loss_fn(expert_weights)

      l_total = l_lm + args.lambda_ortho * total_ortho_loss
      scaled_loss = l_total / args.grad_accum_steps
      scaled_loss.backward()

      if step % args.grad_accum_steps == 0 or step == len(train_loader):
        # 实验 5 动量平滑：在更新前捕获可训练专家参数的历史权重快照
        prev_expert_params = {}
        if 0.0 < args.momentum_beta < 1.0:
          with torch.no_grad():
            for m_idx, moe_layer in enumerate(moe_layers):
              for e_idx, expert in enumerate(moe_layer.experts):
                for p_name, p in expert.named_parameters():
                  if p.requires_grad:
                    prev_expert_params[(m_idx, e_idx, p_name)] = (
                        p.data.detach().clone()
                    )

        optimizer.step()

        # 实验 5 动量平滑：应用演化动量公式更新当前权重
        # theta = theta_new * (1 - beta) + theta_old * beta
        if 0.0 < args.momentum_beta < 1.0 and prev_expert_params:
          with torch.no_grad():
            for m_idx, moe_layer in enumerate(moe_layers):
              for e_idx, expert in enumerate(moe_layer.experts):
                for p_name, p in expert.named_parameters():
                  if (m_idx, e_idx, p_name) in prev_expert_params:
                    old_data = prev_expert_params[(m_idx, e_idx, p_name)]
                    p.data.lerp_(old_data, args.momentum_beta)

        optimizer.zero_grad()

      clear_global_modality_mask()
      total_epoch_loss += l_total.item()

      if step % 10 == 0 or step == len(train_loader):
        print(
            f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}] | "
            f"L_LM: {l_lm.item():.4f} | L_ortho: {float(total_ortho_loss):.4f} | L_total: {l_total.item():.4f}",
            flush=True,
        )

    avg_loss = total_epoch_loss / len(train_loader)
    print(f"🎉 Epoch [{epoch}/{args.epochs}] 完成! 平均 Loss: {avg_loss:.4f}")

    epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
    model.save_pretrained(epoch_dir)
    print(f"💾 Checkpoint 已保存至: {epoch_dir}\n")


if __name__ == "__main__":
  args = parse_args()
  train(args)