# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import transformers
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

from src import conversation as conversation_lib
from src.datasets import make_data_module
from src.model import *
from src.train.mplug_owl2_trainer import MPLUGOwl2Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_type: str = "single"  # [single, pair]
    data_paths: List[str] = field(default_factory=lambda: [])
    data_weights: List[int] = field(default_factory=lambda: [])
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    level_prefix: str = field(default="")
    level_names: List[str] = field(default_factory=lambda: [])
    weight_desp: float = field(default=1.0, metadata={"help": "Absolute weight of description loss."})
    weight_rank: float = field(default=1.0, metadata={"help": "Absolute weight of ranking loss."})
    softkl_loss: bool = field(
        default=False,
        metadata={
            "help": "If True, use softkl_loss for level token; else, use next token loss."
        },
    )
    weight_softkl: float = field(
        default=1.0,
        metadata={
            "help": "Relative weight of softkl loss (w.r.t weight of next token loss as 1.0)."
        },
    )
    weight_next_token: float = field(default=1.0, metadata={"help": "Absolute weight of next token loss."})
    weight_in_level: float = field(default=None, metadata={"help": "Absolute weight of in level loss."})
    continuous_rating_loss: bool = field(
        default=True,
        metadata={
            "help": "Used in pair dataset. If True, use continuous_rating_loss; else, use binary_rating_loss.",
        },
    )
    binary_rating_loss: str = field(
        default="fidelity",
        metadata={
            "help": "Used in pair dataset if continuous_rating_loss is False or no std in dataset. bce loss / fidelity loss.",
            "choices": ["bce", "fidelity"],
        },
    )
    closeset_rating_loss: bool = field(
        default=False,
        metadata={
            "help": "Used in pair dataset. If True, softmax in closeset; else, softmax in openset."
        },
    )
    use_fix_std: bool = field(
        default=True,
        metadata={
            "help": "Use fixed std or predicted std."
        },
    )
    detach_pred_std: bool = field(
        default=False,
        metadata={
            "help": "Detach predicted std."
        },
    )
    tune_visual_abstractor: bool = field(default=False)
    freeze_vision_model: bool = field(default=True)

    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    visual_abstractor_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    save_safetensors: bool = False


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def find_all_lora_names(model):
    lora_module_names = set()
    multimodal_keywords = ["vision_model", "visual_abstractor"]
    for name, _ in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if "v_proj.multiway.1" in name or "q_proj" in name:
            lora_module_names.add(name)

    ls = list(lora_module_names)
    print(ls)
    return ls


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer,
    output_dir: str,
):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()

    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict

        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                # device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=compute_dtype,
        **bnb_model_from_pretrained_args,
    )
    print(model.config)
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_lora_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")

        model = get_peft_model(model, lora_config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            model_args.version
        ]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1"
        ]

    if not training_args.freeze_vision_model and training_args.bits in [4, 8]:
        model.get_model().vision_model.to(
            dtype=compute_dtype, device=training_args.device
        )
    else:
        vision_tower = model.get_model().vision_model
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

    if training_args.tune_visual_abstractor and training_args.bits in [4, 8]:
        model.get_model().visual_abstractor.to(
            dtype=compute_dtype, device=training_args.device
        )
    else:
        visual_abstractor = model.get_model().visual_abstractor
        visual_abstractor.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

    data_args.image_processor = CLIPImageProcessor.from_pretrained(
        model_args.model_name_or_path
    )
    data_args.is_multimodal = True

    model.config.softkl_loss = training_args.softkl_loss
    model.config.weight_desp = training_args.weight_desp
    model.config.weight_next_token = training_args.weight_next_token

    if data_args.dataset_type == "pair":
        model.config.weight_rank = training_args.weight_rank
        model.config.weight_in_level = training_args.weight_in_level
        model.config.continuous_rating_loss = training_args.continuous_rating_loss
        model.config.binary_rating_loss = training_args.binary_rating_loss
        model.config.closeset_rating_loss = training_args.closeset_rating_loss
        model.config.use_fix_std = training_args.use_fix_std
        model.config.detach_pred_std = training_args.detach_pred_std

    if training_args.level_prefix and training_args.level_names:
        model.config.weight_softkl = training_args.weight_softkl
        model.config.level_prefix = tokenizer(training_args.level_prefix).input_ids[1:]
        # index 1: no need start token
        for level_name in training_args.level_names:
            level_id = tokenizer(level_name)["input_ids"]
            assert len(level_id) == 2 and level_id[0] == 1
        model.config.level_ids = [
            id_[1] for id_ in tokenizer(training_args.level_names).input_ids
        ]  # index 1: no need start token
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    for n, p in model.named_parameters():
        if training_args.lora_enable:
            p.requires_grad = True if "lora_" in n else False
            # if "lm_head" in n:
            #     print(n)
            #     p.requires_grad = True
        else:
            p.requires_grad = True
    if training_args.lora_enable:
        model.print_trainable_parameters()

    model.config.tune_visual_abstractor = model_args.tune_visual_abstractor = (
        training_args.tune_visual_abstractor
    )
    print(training_args.tune_visual_abstractor)
    model.get_model().visual_abstractor.requires_grad_(False)
    if training_args.tune_visual_abstractor:
        for n, p in model.get_model().visual_abstractor.named_parameters():
            p.requires_grad = True

    model.config.freeze_vision_model = training_args.freeze_vision_model
    print(training_args.freeze_vision_model)
    model.get_model().vision_model.requires_grad_(True)
    if training_args.freeze_vision_model:
        for p in model.get_model().vision_model.parameters():
            p.requires_grad = False

    if training_args.lora_enable:
        model.print_trainable_parameters()
    model.config.visual_abstractor_lr = training_args.visual_abstractor_lr

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = MPLUGOwl2Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()

    # TODO I dont like auto resume << REMOVE IT AND UNCOMMENT THE ABOVE CODE
    trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir,
        )


if __name__ == "__main__":
    train()
