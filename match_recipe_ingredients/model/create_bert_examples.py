import os
import json

# helper function to convert raw examples into good format for bert
def convert_raw_examples(raw_examples):
    return [
        [
            raw_example["original"],
            raw_example["ingredients"][0]["ingredient"] and raw_example["ingredients"][0]["ingredient"]["id"] or "0"
        ]
        for raw_example in raw_examples
    ]

with open(os.path.join(os.path.dirname(__file__), "data/trainMatchedTrainingExamples.json")) as training_examples_data:
    raw_training_examples = json.load(training_examples_data)

with open(os.path.join(os.path.dirname(__file__), "data/devMatchedTrainingExamples.json")) as dev_examples_data:
    raw_dev_examples = json.load(dev_examples_data)


train_examples = convert_raw_examples(raw_training_examples)

dev_examples = convert_raw_examples(raw_dev_examples)

with open("../../bert_tests/match_recipe_ingredients/data/proprietary/training_examples.json", "w") as f:
    json.dump(train_examples, f, indent=4)

with open("../../bert_tests/match_recipe_ingredients/data/proprietary/dev_examples.json", "w") as f:
    json.dump(dev_examples, f, indent=4)
