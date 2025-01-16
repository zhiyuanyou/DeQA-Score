from dataclasses import dataclass, field
from typing import List, Optional

import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

from src.datasets import make_data_module


@dataclass
class DataArguments:
    dataset_type: str = "pair"
    data_paths: List[str] = field(default_factory=lambda: [])
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)


if __name__ == "__main__":
    cfg_path = "./preprocessor"
    tokenizer = AutoTokenizer.from_pretrained(cfg_path, use_fast=False)
    parser = transformers.HfArgumentParser(DataArguments)
    (data_args,) = parser.parse_args_into_dataclasses()

    data_args.image_folder = "../Data-DeQA-Score/"
    data_args.data_paths = [
        "../Data-DeQA-Score//KONIQ/metas/train_koniq_7k.json",
        "../Data-DeQA-Score//SPAQ/metas/train_spaq_9k.json",
        "../Data-DeQA-Score//KADID10K/metas/train_kadid_8k.json",
    ]
    data_args.data_weights = [1,1,1]
    data_args.image_processor = CLIPImageProcessor.from_pretrained(cfg_path)
    data_args.is_multimodal = True
    data_module = make_data_module(tokenizer=tokenizer, data_args=data_args)
    train_dataset = data_module["train_dataset"]
    collate_fn = data_module["data_collator"]
    data_loader = DataLoader(
        train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True
    )

    for idx, data in enumerate(data_loader):
        print("=" * 100)
        print(f"{idx} / {len(data_loader)}")
        print(data.keys())
        print(data["item_A"].keys())
        print(data["item_B"].keys())
        print(data["item_A"]["image_files"])
        print(data["item_A"]["gt_scores"])
        print(data["item_B"]["image_files"])
        print(data["item_B"]["gt_scores"])
