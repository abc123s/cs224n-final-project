import json

from preprocess_manual import load_examples
from tokenizer import IngredientPhraseTokenizer

ingredientPhraseTokenizer = IngredientPhraseTokenizer()

train_examples = load_examples("./data", "manually_tagged_train")
dev_examples = load_examples("./data", "manually_tagged_dev")

tokenized_train_examples = [
    [ingredientPhraseTokenizer.tokenize(train_example[0]), train_example[1]]
    for train_example in train_examples
]

tokenized_dev_examples = [
    [ingredientPhraseTokenizer.tokenize(dev_example[0]), dev_example[1]]
    for dev_example in dev_examples
]


with open("../bert_tests/extract_recipe_ingredients/data/training_examples.json", "w") as f:
    json.dump(tokenized_train_examples, f, indent=4)

with open("../bert_tests/extract_recipe_ingredients/data/dev_examples.json", "w") as f:
    json.dump(tokenized_dev_examples, f, indent=4)
