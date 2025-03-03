import argparse
import json
import numpy as np
import os
import random
from scipy.stats import norm, pearsonr, spearmanr


def parse_args():
    parser = argparse.ArgumentParser(description="label parameters for DeQA-Score")
    parser.add_argument("--config", type=str, default="./config.json")
    args = parser.parse_args()
    return args


questions = [
    "What do you think about the quality of this image?",
    "Can you rate the quality of this picture?",
    "Can you judge the quality of this image?",
    "How would you rate the quality of this image?",
    "How would you judge the quality of this image?",
    "What is your quality rating for this image?",
    "What's your opinion on the quality of this picture?",
    "Rate the quality of this image.",
    "Could you evaluate the quality of this image?",
    "How do you assess the quality of this image?",
]


def calculate_srcc_plcc(pred, mos):
    srcc, _ = spearmanr(pred, mos)
    plcc, _ = pearsonr(pred, mos)
    return srcc, plcc


def get_level(mos, min_mos, max_mos):
    eps = 1e-8
    texts = ["bad", "poor", "fair", "good", "excellent"]
    for idx in range(1, len(texts) + 1):
        mos_left = min_mos + (idx - 1) / 5 * (max_mos - min_mos) - eps
        mos_right = min_mos + idx / 5 * (max_mos - min_mos) + eps
        if mos > mos_left and mos <= mos_right:
            level = idx
            break
    text = texts[level - 1]
    return text


def adjust_gaussian_bar(probs, score):
    """
    alpha * (a + b + c + d + e) + 5 * beta = 1
    alpha * (5a + 4b + 3c + 2d + e) + 15 beta = score
    ==>
    alpha * A + 5 * beta = 1
    alpha * B + 15 * beta = score
    """
    A = np.array(probs).sum()
    B = np.inner(np.array(probs), np.array([5, 4, 3, 2, 1]))
    alpha = (score - 3) / (B - 3. * A + 1e-9)
    beta = (1. - alpha * A) / 5.
    return alpha, beta


def get_binary_probs(mos, min_mos=1.0, max_mos=5.0):
    eps = 1e-8
    probs = [0, 0, 0, 0, 0]
    for idx in range(1, len(probs)):
        mos_left = min_mos + (idx - 1) / 4 * (max_mos - min_mos) - eps
        mos_right = min_mos + idx / 4 * (max_mos - min_mos) + eps
        if mos > mos_left and mos <= mos_right:
            probs[idx - 1] = (mos_right - mos) / (mos_right - mos_left)
            probs[idx] = (mos - mos_left) / (mos_right - mos_left)
            break
    assert np.array((np.array(probs) == 0)).sum() == 3
    assert round(np.array(probs).sum(), 5) == 1
    probs = probs[::-1]  # should start with "excellent" & end with "bad"
    return probs


def main(cfg):
    density_type = cfg["density_type"]  # ["pdf", "cdf"]
    thre_std = cfg["thre_std"]
    thre_diff = cfg["thre_diff"]
    with open(cfg["split_json"]) as fr:
        split = json.load(fr)
    with open(cfg["mos_json"]) as fr:
        mos_dict = json.load(fr)
    save_train = cfg["save_train"]
    save_test = cfg["save_test"]
    img_dir = cfg["img_dir"]

    moses, stds, imgs = [], [], []
    for img in mos_dict:
        moses.append(mos_dict[img]["mos"])
        stds.append(mos_dict[img]["std"])
        imgs.append(img)
    max_mos = max([float(_) for _ in moses])
    min_mos = min([float(_) for _ in moses])

    num_binary, idx = 0, 0
    preds, gts, raw_diffs, diffs, alphas, betas = [], [], [], [], [], []
    train_metas, test_metas = [], []
    for img, mos_str, std_str in zip(imgs, moses, stds):
        mos, std = float(mos_str), float(std_str)
        if os.path.basename(img) in split["train"]:
            training = True
        elif os.path.basename(img) in split["test"]:
            training = False
        else:
            idx += 1
            # print(idx, img)
            continue

        text = get_level(mos, min_mos, max_mos)
        query = random.choice(questions)
        resp = answer.replace("{}", text)

        # norm mos and std
        mos_norm = 4 * (mos - min_mos) / (max_mos - min_mos) + 1  # [0, 1] -> [1, 5]
        std_norm = 4 * std / (max_mos - min_mos)

        # ["excellent", "good", "fair", "poor", "bad"] -> [5, 4, 3, 2, 1]
        probs = []
        for x in range(5, 0, -1):
            if density_type == "cdf":
                # better for smaller std dataset (see Appendix) like SPAQ
                prob = norm.cdf(x+0.5, mos_norm, std_norm) - norm.cdf(x-0.5, mos_norm, std_norm)
            else:
                # better for larger std dataset (see Appendix) like KonIQ and KADID
                assert density_type == "pdf"
                prob = norm.pdf(x, loc=mos_norm, scale=std_norm)
            probs.append(prob)

        mos_rec = np.inner(np.array(probs), np.array([5, 4, 3, 2, 1]))
        raw_diff = abs(mos_rec - mos_norm)
        raw_diffs.append(raw_diff)

        alpha, beta = adjust_gaussian_bar(probs, mos_norm)
        probs_norm = [max(_ * alpha + beta, 0) for _ in probs]
        mos_rec = np.inner(np.array(probs_norm), np.array([5, 4, 3, 2, 1]))
        diff = abs(mos_rec - mos_norm)

        if std_norm < thre_std or diff > thre_diff:
            # if std is too small, use binary probs (see Appendix)
            probs_norm = get_binary_probs(mos_norm)
            mos_rec = np.inner(np.array(probs_norm), np.array([5, 4, 3, 2, 1]))
            diff, alpha, beta = abs(mos_rec - mos_norm), 1., 0.
            num_binary += 1

        preds.append(mos_rec)
        gts.append(mos_norm)
        diffs.append(diff)
        alphas.append(alpha)
        betas.append(beta)

        meta = {
            "id": os.path.basename(img) + f"->{mos_str}",
            "image": os.path.join(img_dir, img),
            "gt_score": mos,
            "gt_score_norm": mos_norm,
            "level_probs_org": probs,
            "level_probs": probs_norm,
            "std": std,
            "std_norm": std_norm,
        }
        if training:
            conversations = [
                {
                    "from": "human",
                    "value": query + "\n<|image|>",
                },
                {
                    "from": "gpt",
                    "value": resp,
                },
            ]
            meta["conversations"] = conversations
            train_metas.append(meta)
        else:
            del meta["level_probs_org"]
            del meta["level_probs"]
            test_metas.append(meta)

    print("=" * 100)
    print(f"save {len(train_metas)} into {save_train}")
    with open(save_train, "w") as fw:
        fw.write(json.dumps(train_metas, indent=4))

    print(f"save {len(test_metas)} into {save_test}")
    with open(save_test, "w") as fw:
        fw.write(json.dumps(test_metas, indent=4))

    srcc, plcc = calculate_srcc_plcc(preds, gts)
    print("srcc:", srcc, "plcc:", plcc)
    print("[raw_diff]", "l1:", sum(raw_diffs) / len(raw_diffs), "l2:", np.sqrt((np.array(raw_diffs)**2).mean()))
    print("[diff]", "l1:", sum(diffs) / len(diffs), "l2:", np.sqrt((np.array(diffs)**2).mean()))
    print("[alpha]", "mean:", np.mean(alphas), "std:", np.std(alphas))
    print("[beta]", "mean:", np.mean(betas), "std:", np.std(betas))
    print("binary / all:", num_binary, "/", len(train_metas) + len(test_metas))


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as fr:
        cfg = json.load(fr)
    answer = cfg["answer"]
    for dataset in cfg["dataset_params"]:
        random.seed(131)
        main(cfg["dataset_params"][dataset])
