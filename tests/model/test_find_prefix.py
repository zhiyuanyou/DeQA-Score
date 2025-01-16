import torch
from transformers import AutoTokenizer


def find_prefix(input_ids, prefix):
    """
    input_ids: [B, N1], no start token
    prefix: [N2, ], no start token
    """
    len_prefix = prefix.shape[0]  # N2
    # Create all possible windows of len_prefix
    input_ids_unfold = input_ids.unfold(1, len_prefix, 1)
    # Check if all elements in the window match the sequence
    matches = (input_ids_unfold == prefix).all(dim=2)
    # Convert boolean matches to integers for argmax operation
    matches_int = matches.type(torch.int64)
    # Calculate indices for the first match, if any, otherwise set to -1
    indices = torch.where(
        matches.any(dim=1),
        matches_int.argmax(dim=1),
        torch.tensor(-1, dtype=torch.int64),
    )
    return indices


pretrained = "./preprocessor"
tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)

# Example input_ids, [B, N1]
input_ids = tokenizer(
    [
        "I am happy. The quality of the image is good.",
        "ABCD. The quality of the image is poor.",
        "Please go to school. The quality of the image is excellent.",
    ],
    return_tensors="pt",
    padding=True,
).input_ids[:, 1:]
print("=" * 100)
print("input_ids: ")
print(input_ids)

# Example prefix, [N2, ]
prefix = tokenizer("The quality of the image is", return_tensors="pt").input_ids[0, 1:]
print("=" * 100)
print("prefix: ")
print(prefix)

# Find prefix indices
indices = find_prefix(input_ids, prefix)
print("=" * 100)
print("Indices of the prefix: ")
print(indices)
