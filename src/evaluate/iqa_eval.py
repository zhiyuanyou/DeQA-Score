import argparse
import json
import os
from collections import defaultdict
from io import BytesIO

import requests
import torch
from PIL import Image
from tqdm import tqdm

from src.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from src.conversation import conv_templates
from src.mm_utils import get_model_name_from_path, tokenizer_image_token
from src.model.builder import load_pretrained_model


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
        preprocessor_path=args.preprocessor_path,
    )

    meta_paths = args.meta_paths
    root_dir = args.root_dir
    batch_size = args.batch_size
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    with_prob = args.with_prob

    conv_mode = "mplug_owl2"
    inp = "How would you rate the quality of this image?"

    conv = conv_templates[conv_mode].copy()
    inp = inp + "\n" + DEFAULT_IMAGE_TOKEN
    conv.append_message(conv.roles[0], inp)
    image = None

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " The quality of the image is"

    toks = args.level_names
    print(toks)
    ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
    print(ids_)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(args.device)
    )

    for meta_path in meta_paths:
        with open(meta_path) as f:
            iqadata = json.load(f)

        image_tensors = []
        batch_data = []

        imgs_handled = []
        save_path = os.path.join(save_dir, os.path.basename(meta_path))
        if os.path.exists(save_path):
            with open(save_path) as fr:
                for line in fr:
                    meta_res = json.loads(line)
                    imgs_handled.append(meta_res["image"])

        meta_name = os.path.basename(meta_path)
        for i, llddata in enumerate(tqdm(iqadata, desc=f"Evaluating [{meta_name}]")):
            try:
                filename = llddata["image"]
            except:
                filename = llddata["img_path"]
            if filename in imgs_handled:
                continue

            llddata["logits"] = defaultdict(float)
            llddata["probs"] = defaultdict(float)

            image = load_image(os.path.join(root_dir, filename))

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
            image_tensor = (
                image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
                .half()
                .to(args.device)
            )

            image_tensors.append(image_tensor)
            batch_data.append(llddata)

            if (i + 1) % batch_size == 0 or i == len(iqadata) - 1:
                with torch.inference_mode():
                    output_logits = model(
                        input_ids=input_ids.repeat(len(image_tensors), 1),
                        images=torch.cat(image_tensors, 0),
                    )["logits"][:, -1]
                if with_prob:
                    output_probs = torch.softmax(output_logits, dim=1)

                for j, xllddata in enumerate(batch_data):
                    for tok, id_ in zip(toks, ids_):
                        xllddata["logits"][tok] += output_logits[j, id_].item()
                        if with_prob:
                            xllddata["probs"][tok] += output_probs[j, id_].item()
                    meta_res = {
                        "id": xllddata["id"],
                        "image": xllddata["image"],
                        "gt_score": xllddata["gt_score"],
                        "logits": xllddata["logits"],
                    }
                    if with_prob:
                        meta_res["probs"] = xllddata["probs"]
                    with open(save_path, "a") as fw:
                        fw.write(json.dumps(meta_res) + "\n")

                image_tensors = []
                batch_data = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--preprocessor-path", type=str, default=None)
    parser.add_argument("--meta-paths", type=str, required=True, nargs="+")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--level-names", type=str, required=True, nargs="+")
    parser.add_argument("--with-prob", type=bool, default=False)  # whether to save openset prob
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")
    args = parser.parse_args()
    main(args)
