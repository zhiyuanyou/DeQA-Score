import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

from src.constants import IGNORE_INDEX

from .utils import (expand2square, load_video, preprocess,
                    preprocess_multimodal, rank0_print)


class SingleDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_paths: str,
        data_weights: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
    ):
        super(SingleDataset, self).__init__()
        list_data_dict = []
        for data_path, data_weight in zip(data_paths, data_weights):
            data_dict = json.load(open(data_path, "r"))
            list_data_dict += data_dict * data_weight

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def next_rand(self):
        import random

        return random.randint(0, len(self) - 1)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        while True:
            try:
                sources = self.list_data_dict[i]
                if isinstance(i, int):
                    sources = [sources]
                sources_org = copy.deepcopy(sources)
                assert (
                    len(sources) == 1
                ), "Don't know why it is wrapped to a list"  # FIXME
                if "image" in sources_org[0]:
                    image_file = sources_org[0]["image"]

                    image_folder = self.data_args.image_folder
                    processor = self.data_args.image_processor
                    from pathlib import Path

                    # if not Path(os.path.join(image_folder, image_file)).exists():
                    #    i = self.next_rand()
                    #    continue
                    if isinstance(image_file, list):
                        # Multiple Images as Input
                        try:
                            image = [
                                Image.open(os.path.join(image_folder, imfile)).convert(
                                    "RGB"
                                )
                                for imfile in image_file
                            ]
                        except Exception as ex:
                            print(ex)
                            i = self.next_rand()
                            continue
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
                        try:
                            image = Image.open(
                                os.path.join(image_folder, image_file)
                            ).convert("RGB")
                        except Exception as ex:
                            print(ex)
                            i = self.next_rand()
                            continue
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

                    sources = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess(
                    sources,
                    self.tokenizer,
                    has_image=("image" in sources_org[0]),
                )
                if isinstance(i, int):
                    data_dict = dict(
                        input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0],
                    )

                # default task_type: "score", level_probs: [-10000] * 5
                data_dict["task_type"] = sources_org[0].get("task_type", "score")
                data_dict["level_probs"] = sources_org[0].get("level_probs", [-10000] * 5)

                # image exist in the data
                if "image" in sources_org[0]:
                    data_dict["image"] = image
                elif self.data_args.is_multimodal:
                    # image does not exist in the data, but the model is multimodal
                    crop_size = self.data_args.image_processor.crop_size
                    data_dict["image"] = torch.zeros(
                        3, crop_size["height"], crop_size["width"]
                    )
                return data_dict
            except Exception as ex:
                print(ex)
                i = self.next_rand()
                continue


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
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
            input_type="single",
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        batch["task_types"] = [instance["task_type"] for instance in instances]
        batch["level_probs"] = torch.tensor([instance["level_probs"] for instance in instances])

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


def make_single_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SingleDataset(
        tokenizer=tokenizer,
        data_paths=data_args.data_paths,
        data_weights=data_args.data_weights,
        data_args=data_args,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )
