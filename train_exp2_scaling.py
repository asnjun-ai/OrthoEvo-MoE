import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from loss import OrthogonalPenaltyLoss
from patch_model import convert_mllm_to_ortho_evomoe

def parse_args():
    parser = argparse.ArgumentParser(description="Exp 2: Dual-Backbone Scaling")
    parser.add_argument("--model_id", type=str, required=True, help="底座模型标识符")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lambda_ortho", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"📦 正在加载底座模型: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True
    )

    # 适配不同模型的层数结构 (替换深层后 4 层)
    total_layers = len(model.model.layers) if hasattr(model, "model") else len(model.transformer.h)
    target_layers = list(range(total_layers - 4, total_layers))
    print(f"🔧 替换 MoE 目标层索引: {target_layers}")

    model, moe_layers = convert_mllm_to_ortho_evomoe(
        model,
        target_layer_indices=target_layers,
        num_vision_experts=2,
        num_text_experts=2,
        top_k=1
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    for param in model.parameters():
        param.requires_grad = False

    # 针对不同底座映射 LoRA 目标模块
    target_modules = ["w1", "w2", "vision_gate", "text_gate"]
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # 模拟微调集 (多模态投射与多任务推理问答)
    dummy_sentences = [
        "Visual feature orthogonal decomposition enforces functional divergence.",
        "Modality isolation separates vision token representation from textual context.",
        "Mixture of experts scales parameter efficiency in dense language backbones.",
        "Routing entropy remains stable across architectural scaling regimes."
    ] * 5

    inputs = tokenizer(dummy_sentences, return_tensors="pt", padding=True, truncation=True, max_length=64)
    dataset = torch.utils.data.TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    ortho_loss_fn = OrthogonalPenaltyLoss(sample_rows=128)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    print(f"🚀 开始训练底座 {args.model_id} | λ_ortho = {args.lambda_ortho}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (b_input_ids, b_attn_mask) in enumerate(dataloader, start=1):
            b_input_ids = b_input_ids.to(device)
            b_attn_mask = b_attn_mask.to(device)

            outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels=b_input_ids)
            l_lm = outputs.loss

            total_ortho_loss = 0.0
            if args.lambda_ortho > 0:
                for moe_layer in moe_layers:
                    total_ortho_loss += ortho_loss_fn(moe_layer.get_expert_weights())

            loss = (l_lm + args.lambda_ortho * total_ortho_loss) / args.grad_accum_steps
            loss.backward()

            if step % args.grad_accum_steps == 0 or step == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum_steps

        print(f"Epoch [{epoch}/{args.epochs}] 完成 | Loss: {total_loss / len(dataloader):.4f}")

    save_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(save_path)
    print(f"💾 Checkpoint 已保存至: {save_path}\n")

if __name__ == "__main__":
    main()