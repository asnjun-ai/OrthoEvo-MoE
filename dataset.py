import os
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from modality_utils import get_modality_mask

class MLLMInstructionDataset(Dataset):
    """
    支持动态长度的多模态 Instruction Dataset
    """
    def __init__(self, json_path, image_dir, processor, max_length=1024):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        conversations = item["conversations"]
        user_prompt = ""
        assistant_response = ""

        for conv in conversations:
            if conv["from"] == "human":
                user_prompt = conv["value"]
            elif conv["from"] == "gpt":
                assistant_response = conv["value"]

        if "<image>" not in user_prompt:
            user_prompt = f"<image>\n{user_prompt}"

        full_text = f"USER: {user_prompt}\nASSISTANT: {assistant_response}"

        image = None
        if "image" in item and item["image"]:
            img_path = os.path.join(self.image_dir, item["image"])
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception:
                    image = None

        if image is None:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        # 💡 不在单样本内部填充，保留自然长度，交给 DataCollator 进行 Batch 动态填充
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)

        # 构建 Labels
        labels = input_ids.clone()
        prompt_text = f"USER: {user_prompt}\nASSISTANT:"
        prompt_ids = self.processor.tokenizer(prompt_text, return_tensors="pt")["input_ids"].squeeze(0)
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels[:prompt_len] = -100

        # 生成 Modality Mask
        modality_mask = get_modality_mask(input_ids.unsqueeze(0), image_token_id=32000).squeeze(0)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "labels": labels,
            "modality_mask": modality_mask
        }


class MLLMDataCollator:
    """
    动态 Batch 整理器：将不同长度的样本动态填充至当前 Batch 的最大长度
    """
    def __init__(self, pad_token_id=0, ignore_index=-100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        modality_mask = [item["modality_mask"] for item in batch]

        # 1. 动态填充 input_ids, labels, modality_mask 到当前 Batch 最大长度
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)
        modality_mask_padded = pad_sequence(modality_mask, batch_first=True, padding_value=0)

        # 2. 图片特征 pixel_values 形状固定 [3, 224, 224]，直接 stack
        pixel_values_stacked = torch.stack(pixel_values, dim=0)

        return {
            "input_ids": input_ids_padded,
            "pixel_values": pixel_values_stacked,
            "labels": labels_padded,
            "modality_mask": modality_mask_padded
        }