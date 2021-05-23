from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./experiments/20210521_0844_afd6b0b",
    tokenizer="./experiments/20210521_0844_afd6b0b"
)

print(fill_mask("1 cup [MASK]."))

print(fill_mask("2 lb [MASK]."))

print(fill_mask("1 [MASK] milk."))
