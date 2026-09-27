import torch
import torch.nn as nn
from moe_layer import OrthoEvoMoELayer


def get_model_layers(model):
    """
    智能搜寻各种 MLLM (LLaVA, Qwen-VL, Qwen2-VL 等) 的 Transformer Layers 路径
    """
    possible_paths = [
        ["language_model", "model", "layers"],  # 标准 LLaVA 1.5/1.6
        ["language_model", "layers"],           # 简化版 LLaVA / Qwen-VL
        ["model", "layers"],                    # 标准 LLaMA / Qwen 底座
        ["model", "language_model", "layers"],
        ["text_model", "encoder", "layers"],
    ]

    for path in possible_paths:
        curr = model
        found = True
        for attr in path:
            if hasattr(curr, attr):
                curr = getattr(curr, attr)
            else:
                found = False
                break
        if found and isinstance(curr, (torch.nn.ModuleList, list)):
            return curr

    # 兜底策略：遍历查找第一个名为 'layers' 的 ModuleList
    for name, module in model.named_modules():
        if name.endswith(".layers") and isinstance(module, torch.nn.ModuleList):
            return module

    raise AttributeError(
        f"未能自动识别模型架构 {type(model).__name__} 的 Transformer layers 路径！"
    )


def convert_mllm_to_ortho_evomoe(
    model,
    target_layer_indices=None,
    num_vision_experts=2,
    num_text_experts=2,
    top_k=1,
    router_type="explicit",
):
    """
    将 HuggingFace 多模态模型的指定层 FFN 替换为 OrthoEvoMoELayer
    """
    # 1. 定位 Transformer layers
    layers = get_model_layers(model)

    # 2. 读取隐藏层维度
    text_config = getattr(model.config, "text_config", model.config)
    hidden_dim = getattr(
        text_config, "hidden_size", getattr(model.config, "hidden_size", 4096)
    )
    intermediate_dim = getattr(
        text_config,
        "intermediate_size",
        getattr(model.config, "intermediate_size", hidden_dim * 4),
    )

    if target_layer_indices is None:
        total_layers = len(layers)
        target_layer_indices = list(range(total_layers // 2, total_layers))

    print(
        f"🔧 成功定位到 Transformer 架构 (Hidden Dim: {hidden_dim}, Intermediate Dim: {intermediate_dim})"
    )
    print(f"🔧 正在将以下层替换为 OrthoEvoMoE 专家层: {target_layer_indices}")

    replaced_moe_layers = []

    for idx in target_layer_indices:
        original_mlp = layers[idx].mlp

        # 3. 确定目标设备
        target_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # 4. 在 CPU 初始化 MoE 层
        moe_layer = OrthoEvoMoELayer(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_vision_experts=num_vision_experts,
            num_text_experts=num_text_experts,
            top_k=top_k,
            router_type=router_type,
        )

        # 5. 安全热启动：兼容 4-bit 量化权重 (Linear4bit) 与 标准浮点权重
        with torch.no_grad():
            for expert in moe_layer.experts:
                proj_w1 = getattr(
                    original_mlp, "gate_proj", getattr(original_mlp, "w1", None)
                )
                proj_w2 = getattr(
                    original_mlp, "down_proj", getattr(original_mlp, "w2", None)
                )

                if proj_w1 is not None and proj_w2 is not None:
                    # 情况 A: 底座为 4-bit 量化层 (Linear4bit)
                    if hasattr(proj_w1, "quant_state"):
                        try:
                            import bitsandbytes.functional as bnb_F

                            decomp_w1 = bnb_F.dequantize_4bit(
                                proj_w1.weight.data, proj_w1.weight.quant_state
                            )
                            decomp_w2 = bnb_F.dequantize_4bit(
                                proj_w2.weight.data, proj_w2.weight.quant_state
                            )
                            expert.w1.weight.copy_(decomp_w1.to(dtype=torch.float16, device="cpu"))
                            expert.w2.weight.copy_(decomp_w2.to(dtype=torch.float16, device="cpu"))
                        except Exception as e:
                            print(f"⚠️ 4-bit 权重反量化跳过，使用标准初始化: {e}")
                    # 情况 B: 标准非量化 Float 权重
                    elif (
                        hasattr(proj_w1, "weight")
                        and proj_w1.weight.device.type != "meta"
                    ):
                        if proj_w1.weight.shape == expert.w1.weight.shape:
                            expert.w1.weight.copy_(
                                proj_w1.weight.data.to(dtype=torch.float16, device="cpu")
                            )
                            expert.w2.weight.copy_(
                                proj_w2.weight.data.to(dtype=torch.float16, device="cpu")
                            )

        # 6. 一次性转移到 GPU 并挂载替换
        moe_layer = moe_layer.to(device=target_device, dtype=torch.float16)
        layers[idx].mlp = moe_layer
        replaced_moe_layers.append(moe_layer)

    print(f"✅ 成功替换 {len(replaced_moe_layers)} 个 FFN 层为 OrthoEvoMoE 专家层！")
    return model, replaced_moe_layers