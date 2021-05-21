from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./experiments/20210520_1824_1659fe0",
    tokenizer="./experiments/20210520_1824_1659fe0"
)

print(fill_mask("1 cup [MASK]."))

print(fill_mask("2 lb [MASK]."))