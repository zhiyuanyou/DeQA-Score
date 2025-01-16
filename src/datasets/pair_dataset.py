import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

from src.constants import IGNORE_INDEX

from .utils import (expand2square, load_video, preprocess,
                    preprocess_multimodal, rank0_print)


class PairDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_paths,
        data_weights,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
    ):
        super(PairDataset, self).__init__()
        dataset_list = []  # list (different datasets) of list (samples in one dataset)
        for data_path, data_weight in zip(data_paths, data_weights):
            data_list = json.load(open(data_path, "r"))
            dataset_list.append(data_list * data_weight)
        self.dataset_list = dataset_list

        # Construct nums_data, nums_data[i] is the number of samples in 0-i th datasets
        nums_eachdata = [len(_) for _ in self.dataset_list]
        nums_predata = copy.deepcopy(nums_eachdata)
        for idx in range(1, len(nums_predata)):
            nums_predata[idx] = nums_predata[idx] + nums_predata[idx - 1]

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.nums_eachdata = nums_eachdata
        self.nums_predata = nums_predata
        self.data_args = data_args
        assert self.nums_predata[-1] == sum(self.nums_eachdata)

    def __len__(self):
        return self.nums_predata[-1]

    @property
    def lengths(self):
        length_list = []
        for dataset in self.dataset_list:
            for sample in dataset:
                img_tokens = 128 if "image" in sample else 0
                length_list.append(
                    sum(len(conv["value"].split()) for conv in sample["conversations"])
                    + img_tokens
                )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for dataset in self.dataset_list:
            for sample in dataset:
                cur_len = sum(
                    len(conv["value"].split()) for conv in sample["conversations"]
                )
                cur_len = cur_len if "image" in sample else -cur_len
                length_list.append(cur_len)
        return length_list

    def next_rand(self):
        return random.randint(0, len(self) - 1)

    def __getitem__(self, i):
        while True:
            try:
                # Get idx_dataset, idx_sample
                if i < self.nums_predata[0]:
                    idx_dataset = 0
                    idx_sample = i
                else:
                    for idx_dataset in range(1, len(self.nums_predata)):
                        if (
                            i < self.nums_predata[idx_dataset]
                            and i >= self.nums_predata[idx_dataset - 1]
                        ):
                            idx_sample = i - self.nums_predata[idx_dataset - 1]
                            break
                # Sample two items
                item_A = self.get_one_item(idx_dataset, idx_sample)
                while True:
                    idx_sample_B = random.randint(
                        0, self.nums_eachdata[idx_dataset] - 1
                    )
                    if idx_sample_B != idx_sample:
                        break
                item_B = self.get_one_item(idx_dataset, idx_sample_B)
                return {
                    "item_A": item_A,
                    "item_B": item_B,
                }
            except Exception as ex:
                print(ex)
                i = self.next_rand()
                continue

    def get_one_item(self, idx_dataset, idx_sample) -> Dict[str, torch.Tensor]:
        # For IQA data, i must be int
        sources = [self.dataset_list[idx_dataset][idx_sample]]
        sources_org = copy.deepcopy(sources)
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources_org[0]:
            image_file = sources[0]["image"]

            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor

            if isinstance(image_file, list):
                # Multiple Images as Input
                image = [
                    Image.open(os.path.join(image_folder, imfile)).convert("RGB")
                    for imfile in image_file
                ]

                if self.data_args.image_aspect_ratio == "pad":
                    image = [
                        expand2square(
                            img,
                            tuple(int(x * 255) for x in processor.image_mean),
                        )
                        for img in image
                    ]
                    image = processor.preprocess(image, return_tensors="pt")[
                        "pixel_values"
                    ]
                else:
                    image = processor.preprocess(image, return_tensors="pt")[
                        "pixel_values"
                    ]
            elif os.path.join(image_folder, image_file).endswith("mp4"):
                # Video as Input
                image = load_video(os.path.join(image_folder, image_file))
                if self.data_args.image_aspect_ratio == "pad":
                    image = [
                        expand2square(
                            img,
                            tuple(int(x * 255) for x in processor.image_mean),
                        )
                        for img in image
                    ]
                    image = processor.preprocess(image, return_tensors="pt")[
                        "pixel_values"
                    ]
                else:
                    image = processor.preprocess(image, return_tensors="pt")[
                        "pixel_values"
                    ]
            else:
                image = Image.open(os.path.join(image_folder, image_file)).convert(
                    "RGB"
                )
                if self.data_args.image_aspect_ratio == "pad":
                    image = expand2square(
                        image, tuple(int(x * 255) for x in processor.image_mean)
                    )
                    image = processor.preprocess(image, return_tensors="pt")[
                        "pixel_values"
                    ]
                else:
                    image = processor.preprocess(image, return_tensors="pt")[
                        "pixel_values"
                    ]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args,
            )
        else:
            # Without images
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=("image" in sources_org[0]),
        )
        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
        )

        # default task_type: "score", gt_socre & std: -10000, level_probs: [-10000] * 5
        data_dict["task_type"] = sources_org[0].get("task_type", "score")
        data_dict["gt_score"] = sources_org[0].get("gt_score", -10000)
        data_dict["std"] = sources_org[0].get("std", -10000)
        data_dict["level_probs"] = sources_org[0].get("level_probs", [-10000] * 5)

        # image exist in the data
        if "image" in sources_org[0]:
            data_dict["image_file"] = image_file
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
        return data_dict


@dataclass
class DataCollatorForPairDataset(object):
    """Collate examples for pair fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances_A = [instance["item_A"] for instance in instances]
        instances_B = [instance["item_B"] for instance in instances]
        batch = {
            "input_type": "pair",
            "item_A": self.collate_one(instances_A),
            "item_B": self.collate_one(instances_B),
        }
        return batch

    def collate_one(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        batch["task_types"] = [instance["task_type"] for instance in instances]
        batch["gt_scores"] = torch.tensor([instance["gt_score"] for instance in instances])
        batch["stds"] = torch.tensor([instance["std"] for instance in instances])
        batch["level_probs"] = torch.tensor([instance["level_probs"] for instance in instances])

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images
            batch["image_files"] = [instance["image_file"] for instance in instances]

        return batch


def make_pair_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = PairDataset(
        tokenizer=tokenizer,
        data_paths=data_args.data_paths,
        data_weights=data_args.data_weights,
        data_args=data_args,
    )
    data_collator = DataCollatorForPairDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )
