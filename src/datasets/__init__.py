from .pair_dataset import make_pair_data_module
from .single_dataset import make_single_data_module


def make_data_module(tokenizer, data_args):
    if data_args.dataset_type == "single":
        return make_single_data_module(tokenizer, data_args)
    elif data_args.dataset_type == "pair":
        return make_pair_data_module(tokenizer, data_args)
    else:
        raise ValueError
