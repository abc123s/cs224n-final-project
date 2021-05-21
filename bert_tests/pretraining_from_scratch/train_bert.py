import os
import subprocess
import json
from datetime import datetime

from tokenizers import BertWordPieceTokenizer

from transformers import (
    BertConfig, 
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset, 
    Trainer,
    TrainingArguments, 
)

params = {
    # tokenizer params
    "LOWERCASED": False,
    "VOCAB_SIZE": 30000, # default value for bert-base is 30k
    "MIN_FREQUENCY": 2, # default value for bert-base is 2
    "LIMIT_ALPHABET": 1000, # default value for bert-base is 2

    # BERT params
    "HIDDEN_SIZE": 768, # default value for bert-base is 768
    "NUM_HIDDEN_LAYERS": 12, # default value for bert-base is 12
    "NUM_ATTENTION_HEADS": 12, # default value for bert-base is 12
    # feed forward layer size
    "INTERMEDIATE_SIZE": 3072, # default value for bert-base is 3072
    "HIDDEN_ACT": "gelu", # default value for bert-base is "gelu"
    "HIDDEN_DROPOUT_PROB": 0.1, # default value for bert-base is 0.1
    "ATTENTION_PROBS_DROPOUT_PROB": 0.1, # default value for bert-base is 0.1
    "MAX_POSITION_EMBEDDINGS": 512, # default value for bert-base is 512
    "TYPE_VOCAB_SIZE": 2, # default value for bert-base is 2
    "INITIALIZER_RANGE": 0.02, # default value for bert-base is 0.02
    "LAYER_NORM_EPS": 1e-12, # default value for bert-base is 1e-12
    "POSITION_EMBEDDING_TYPE": "absolute", # default value for bert-base is "absolute"

    # training data
    "CORPUSES": [
        '../corpuses/ing_ins_rec_doc_cased_tiny.txt',
    ],
    "LINE_BY_LINE": True,
    "MLM_PROBABILITY": 0.15, # default for bert is 0.15

    # training
    "EPOCHS": 3,
    "BATCH_SIZE": 16,
}

# make experiment directory and save experiment params down
date_string = datetime.now().strftime("%Y%m%d_%H%M")
commit_string = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
experiment_dir = "experiments/" + date_string + "_" + commit_string
os.mkdir(experiment_dir)
os.mkdir(experiment_dir + "/logs")
os.mkdir(experiment_dir + "/results")

# save params down
with open(experiment_dir + "/params.json", "w") as f:
    json.dump(params, f, indent=4)

# make empty cased BERT wordpiece tokenizer
tokenizer = BertWordPieceTokenizer(
    lowercase=params["LOWERCASED"],
)

# train tokenizer on corpus
tokenizer.train(
    files=params["CORPUSES"],
    vocab_size=params["VOCAB_SIZE"],
    min_frequency=params["MIN_FREQUENCY"],
    limit_alphabet=params["LIMIT_ALPHABET"],
)

# save tokenizer for later use
tokenizer.save_model(experiment_dir)

# reload tokenizer in the transformers library
tokenizer = BertTokenizerFast.from_pretrained(experiment_dir)

# prepare for MLM training
config = BertConfig(
    vocab_size=params["VOCAB_SIZE"],
    hidden_size=params["HIDDEN_SIZE"],
    num_hidden_layers=params["NUM_HIDDEN_LAYERS"],
    num_attention_heads=params["NUM_ATTENTION_HEADS"],
    # feed forward layer size
    intermediate_size=params["INTERMEDIATE_SIZE"],
    hidden_act=params["HIDDEN_ACT"],
    hidden_dropout_prob=params["HIDDEN_DROPOUT_PROB"],
    attention_probs_dropout_prob=params["ATTENTION_PROBS_DROPOUT_PROB"],
    max_position_embeddings=params["MAX_POSITION_EMBEDDINGS"],
    type_vocab_size=params["TYPE_VOCAB_SIZE"],
    initializer_range=params["INITIALIZER_RANGE"],
    layer_norm_eps=params["LAYER_NORM_EPS"],
    position_embedding_type=params["POSITION_EMBEDDING_TYPE"],
)

if params["LINE_BY_LINE"] and len(params["CORPUSES"]):
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=params["CORPUSES"][0],
        block_size=params["MAX_POSITION_EMBEDDINGS"],
    )
else:
    raise NotImplementedError

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=params["MLM_PROBABILITY"]
)

model = BertForMaskedLM(config=config)

# train
training_args = TrainingArguments(
    output_dir=experiment_dir,
    num_train_epochs=params["EPOCHS"],
    per_gpu_train_batch_size=params["BATCH_SIZE"],
    save_steps=10000,
    save_total_limit=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

# save model
trainer.save_model(experiment_dir)