from transformers import AutoTokenizer

if __name__ == "__main__":
    model_base = "./preprocessor"
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

    print("=" * 100)
    levels = ["minimal", "low", "medium", "high", "severe"]
    for level in levels:
        input_ids = tokenizer(level)["input_ids"]
        assert len(input_ids) == 2 and input_ids[0] == 1
        text = tokenizer.decode(input_ids[1])
        print(input_ids, text)
