import argparse
import json

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="evaluation parameters for DeQA-Score")
    parser.add_argument("--level_names", type=str, required=True, nargs="+")
    parser.add_argument("--pred_paths", type=str, required=True, nargs="+")
    parser.add_argument("--gt_paths", type=str, required=True, nargs="+")
    parser.add_argument("--use_openset_probs", action="store_true")
    args = parser.parse_args()
    return args


def kl_divergence(mu_1, mu_2, sigma_1, sigma_2):
    """
    Calculate the Kullback-Leibler (KL) divergence between two Gaussian distributions for numpy arrays.
    
    Parameters:
    mu_1 (np.array): Mean of the first distribution (array of size N).
    mu_2 (np.array): Mean of the second distribution (array of size N).
    sigma_1 (np.array): Standard deviation of the first distribution (array of size N).
    sigma_2 (np.array): Standard deviation of the second distribution (array of size N).
    
    Returns:
    np.array: KL divergence from distribution 1 to distribution 2 (array of size N).
    """
    eps = 1e-8
    return np.log(sigma_2 / (sigma_1 + eps)) + (sigma_1**2 + (mu_1 - mu_2)**2) / (2 * sigma_2**2 + eps) - 0.5


def js_divergence(mu_1, mu_2, sigma_1, sigma_2):
    """
    Calculate the Jensen-Shannon (JS) divergence between two Gaussian distributions for numpy arrays.
    
    Parameters:
    mu_1 (np.array): Mean of the first distribution (array of size N).
    mu_2 (np.array): Mean of the second distribution (array of size N).
    sigma_1 (np.array): Standard deviation of the first distribution (array of size N).
    sigma_2 (np.array): Standard deviation of the second distribution (array of size N).
    
    Returns:
    np.array: JS divergence between the two distributions (array of size N).
    """
    # Midpoint distribution parameters
    mu_m = 0.5 * (mu_1 + mu_2)
    sigma_m = np.sqrt(0.5 * (sigma_1**2 + sigma_2**2))
    
    # JS divergence as the average of the KL divergences
    return 0.5 * kl_divergence(mu_1, mu_m, sigma_1, sigma_m) + 0.5 * kl_divergence(mu_2, mu_m, sigma_2, sigma_m)


def wasserstein_distance(mu_1, mu_2, sigma_1, sigma_2):
    """
    Calculate the Wasserstein distance between two Gaussian distributions for numpy arrays.
    
    Parameters:
    mu_1 (np.array): Mean of the first distribution (array of size N).
    mu_2 (np.array): Mean of the second distribution (array of size N).
    sigma_1 (np.array): Standard deviation of the first distribution (array of size N).
    sigma_2 (np.array): Standard deviation of the second distribution (array of size N).
    
    Returns:
    np.array: Wasserstein distance between the two distributions (array of size N).
    """
    return np.sqrt((mu_1 - mu_2)**2 + (sigma_1 - sigma_2)**2)


def cal_score(level_names, logits=None, probs=None, use_openset_probs=False):
    if use_openset_probs:
        assert logits is None
        probs = np.array([probs[_] for _ in level_names])
    else:
        assert probs is None
        logprobs = np.array([logits[_] for _ in level_names])
        probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    score = np.inner(probs, np.array([5., 4., 3., 2., 1.]))
    return score, probs


def cal_std(score, probs):
    variance = (np.array([5., 4., 3., 2., 1.]) - score) * (np.array([5., 4., 3., 2., 1.]) - score)
    variance = np.inner(probs, variance)
    std = np.sqrt(variance)
    return std


if __name__ == "__main__":
    args = parse_args()
    level_names = args.level_names
    pred_paths = args.pred_paths
    gt_paths = args.gt_paths
    use_openset_probs = args.use_openset_probs

    for pred_path, gt_path in zip(pred_paths, gt_paths):
        print("=" * 100)
        print("Pred: ", pred_path)
        print("GT: ", gt_path)

        # load predict results
        pred_metas = []
        with open(pred_path) as fr:
            for line in fr:
                pred_meta = json.loads(line)
                pred_metas.append(pred_meta)

        # load gt results
        with open(gt_path) as fr:
            gt_metas = json.load(fr)

        pred_metas.sort(key=lambda x: x["id"])
        gt_metas.sort(key=lambda x: x["id"])

        mu_preds = []
        std_preds = []
        mu_gts = []
        std_gts = []
        for pred_meta, gt_meta in zip(pred_metas, gt_metas):
            assert pred_meta["id"] == gt_meta["id"]
            if use_openset_probs:
                pred_score, probs = cal_score(level_names, logits=pred_meta["logits"], use_openset_probs=True)
            else:
                pred_score, probs = cal_score(level_names, logits=pred_meta["logits"], use_openset_probs=False)
            pred_std = cal_std(pred_score, probs)
            mu_preds.append(pred_score)
            std_preds.append(pred_std)
            mu_gts.append(gt_meta["gt_score_norm"])
            std_gts.append(gt_meta["std_norm"])

        mu_preds = np.array(mu_preds)
        std_preds = np.array(std_preds)
        mu_gts = np.array(mu_gts)
        std_gts = np.array(std_gts)

        kl = kl_divergence(mu_gts, mu_preds, std_gts, std_preds).mean()
        js = js_divergence(mu_gts, mu_preds, std_gts, std_preds).mean()
        wd = wasserstein_distance(mu_gts, mu_preds, std_gts, std_preds).mean()

        print(f"KL: {kl}")
        print(f"JS: {js}")
        print(f"WD: {wd}")
