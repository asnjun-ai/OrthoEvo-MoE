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
        ["text_model", "encoder", "layers"]
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

    raise AttributeError(f"未能自动识别模型架构 {type(model).__name__} 的 Transformer layers 路径！")


def convert_mllm_to_ortho_evomoe(
    model, 
    target_layer_indices=None, 
    num_vision_experts=2, 
    num_text_experts=2, 
    top_k=1
):
    """
    将 HuggingFace 多模态模型的指定层 FFN 替换为 OrthoEvoMoELayer
    """
    # 1. 智能定位 layers
    layers = get_model_layers(model)

    # 2. 智能读取隐藏层维度 (兼容 LLaVA 的 text_config 和标准 config)
    text_config = getattr(model.config, "text_config", model.config)
    hidden_dim = getattr(text_config, "hidden_size", getattr(model.config, "hidden_size", 4096))
    intermediate_dim = getattr(text_config, "intermediate_size", getattr(model.config, "intermediate_size", hidden_dim * 4))

    # 默认替换后半部分层 (如 32 层中的 16~31 层)
    if target_layer_indices is None:
        total_layers = len(layers)
        target_layer_indices = list(range(total_layers // 2, total_layers))

    print(f"🔧 成功定位到 Transformer 架构 (Hidden Dim: {hidden_dim}, Intermediate Dim: {intermediate_dim})")
    print(f"🔧 正在将以下层替换为 OrthoEvoMoE 专家层: {target_layer_indices}")

    replaced_moe_layers = []

    for idx in target_layer_indices:
        original_mlp = layers[idx].mlp

        # 3. 确定目标设备与数据类型 (避免 meta 设备冲突)
        target_device = next(original_mlp.parameters()).device
        target_dtype = next(original_mlp.parameters()).dtype

        if target_device.type == "meta":
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 4. 创建全新的 OrthoEvoMoELayer
        moe_layer = OrthoEvoMoELayer(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_vision_experts=num_vision_experts,
            num_text_experts=num_text_experts,
            top_k=top_k
        ).to(device=target_device, dtype=target_dtype)

        # 5. 热启动权重复制 (用原始 FFN 权重初始化所有 Expert)
        with torch.no_grad():
            for expert in moe_layer.experts:
                if hasattr(original_mlp, "w1") and original_mlp.w1.weight.device.type != "meta":
                    expert.w1.weight.copy_(original_mlp.w1.weight)
                    expert.w2.weight.copy_(original_mlp.w2.weight)
                elif hasattr(original_mlp, "gate_proj") and original_mlp.gate_proj.weight.device.type != "meta":
                    # 兼容 LLaMA/LLaVA 架构里的 gate_proj / down_proj 命名
                    expert.w1.weight.copy_(original_mlp.gate_proj.weight)
                    expert.w2.weight.copy_(original_mlp.down_proj.weight)

        # 6. 执行替换
        layers[idx].mlp = moe_layer
        replaced_moe_layers.append(moe_layer)

    print(f"✅ 成功替换 {len(replaced_moe_layers)} 个 FFN 层为 OrthoEvoMoE 专家层！")
    return model, replaced_moe_layers