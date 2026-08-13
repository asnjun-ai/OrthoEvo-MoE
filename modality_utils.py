import torch

def get_modality_mask(input_ids, image_token_id=32000):
    """
    根据 input_ids 自动构建 0-1 模态掩码:
    1 表示 Vision Token, 0 表示 Text Token
    
    input_ids: [Batch_Size, Seq_Len]
    """
    modality_mask = (input_ids == image_token_id).long()
    return modality_mask