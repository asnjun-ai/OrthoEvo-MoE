import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from loss import OrthogonalPenaltyLoss
from patch_model import convert_mllm_to_ortho_evomoe
from modality_utils import get_modality_mask

def train_mllm_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "llava-hf/llava-1.5-7b-hf"  # 或你的本地 Qwen-VL / LLaVA 路径

    print("1. 正在加载预训练 HuggingFace 多模态大模型...")
    # 实际训练时加载真实模型，此处演示结构挂载
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    # 2. 挂载 OrthoEvoMoELayer (替换第 16 到 31 层)
    model, moe_layers = convert_mllm_to_ortho_evomoe(
        model, 
        target_layer_indices=list(range(16, 32)),
        num_vision_experts=2,
        num_text_experts=2,
        top_k=1
    )

    # 3. 优化器与正交 Loss 准备
    ortho_loss_fn = OrthogonalPenaltyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    lambda_ortho = 0.05

    # 4. 模拟一个 Forward & Backward Step
    # (真实训练中由 DataLoader 提供 input_ids, pixel_values, labels)
    dummy_input_ids = torch.randint(0, 32000, (1, 128), device=device)
    dummy_input_ids[0, 10:50] = 32000  # 假设中间 40 个 Token 是图片 Token
    dummy_labels = dummy_input_ids.clone()

    # 构建 Modality Mask
    modality_mask = get_modality_mask(dummy_input_ids, image_token_id=32000)

    # ------------------ Forward ------------------
    optimizer.zero_grad()

    # MLLM 主前向传播 (获取语言模型自带的 Causal LM Loss)
    outputs = model(input_ids=dummy_input_ids, labels=dummy_labels)
    l_lm = outputs.loss  # 语言模型主交叉熵损失

    # 收集所有挂载的 MoE 层的正交损失 L_ortho
    total_ortho_loss = 0.0
    for moe_layer in moe_layers:
        expert_weights = moe_layer.get_expert_weights()
        total_ortho_loss += ortho_loss_fn(expert_weights)

    # 汇总总 Loss: L_total = L_LM + λ * ∑ L_ortho
    l_total = l_lm + lambda_ortho * total_ortho_loss

    # ------------------ Backward ------------------
    l_total.backward()
    optimizer.step()

    print(f"✅ Step 完成! L_LM: {l_lm.item():.4f} | L_ortho_sum: {total_ortho_loss.item():.4f} | Total: {l_total.item():.4f}")

if __name__ == "__main__":
    train_mllm_demo()