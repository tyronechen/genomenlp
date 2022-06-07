import argparse
import gzip
import os
from random import shuffle
from warnings import warn
from datasets import ClassLabel, Dataset, DatasetDict, Value, load_dataset
import pandas as pd
import screed
import torch
from tokenizers import SentencePieceUnigramTokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, DistilBertConfig, \
    DistilBertModel, Trainer, TrainingArguments

def load_data(infile_path: str):
    """Take a 🤗 dataset object, path as output and write files to disk"""
    if infile_path.endswith(".csv") or infile_path.endswith(".csv.gz"):
        return load_dataset("csv", data_files=infile_path)
    elif infile_path.endswith(".json"):
        return load_dataset("json", data_files=infile_path)
    elif infile_path.endswith(".parquet"):
        return load_dataset("parquet", data_files=infile_path)

def split_datasets(dataset: DatasetDict, train: float, test: float=0,
                   val: float=0, shuffle: bool=False):
    """Split data into training | testing | validation sets"""
    assert train + test + val == 1, "Proportions of datasets must sum to 1!"
    train_split = 1 - train
    test_split = 1 - test / (test + val)
    val_split = 1 - val / (test + val)

    train = dataset.train_test_split(test_size=train_split, shuffle=shuffle)
    if val > 0:
        test_valid = train['test'].train_test_split(test_size=test_split, shuffle=shuffle)
        return DatasetDict({
            'train': train['train'],
            'test': test_valid['test'],
            'valid': test_valid['train'],
            })
    else:
        return DatasetDict({
            'train': train['train'],
            'test': train['test'],
            })

def main():
    parser = argparse.ArgumentParser(
        description='Take HuggingFace🤗  dataset and train.'
    )
    parser.add_argument('infile_path', type=str,
                        help='path to [ csv | csv.gz | json | parquet ] file')
    parser.add_argument('-o', '--outfile', type=str, default="hf_out/",
                        help='write 🤗 dataset to disk as \
                        [ csv | json | parquet | dir/ ] (DEFAULT: "hf_out/")')
    parser.add_argument('-t', '--tokeniser_path', type=str, default="",
                        help='path to tokeniser.json file to load data from')
    parser.add_argument('--split_train', type=float, default=0.90,
                        help='proportion of training data (DEFAULT: 0.90)')
    parser.add_argument('--split_test', type=float, default=0.05,
                        help='proportion of testing data (DEFAULT: 0.05)')
    parser.add_argument('--split_val', type=float, default=0.05,
                        help='proportion of validation data (DEFAULT: 0.05)')
    parser.add_argument('--no_shuffle', action="store_false",
                        help='turn off random shuffling (DEFAULT: Shuffle)')

    args = parser.parse_args()
    infile_path = args.infile_path
    tokeniser_path = args.tokeniser_path
    split_train = args.split_train
    split_test = args.split_test
    split_val = args.split_val
    outfile = args.outfile
    shuffle = args.no_shuffle

    if os.path.exists(tokeniser_path):
        special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
        print("USING EXISTING TOKENISER:", tokeniser_path)
        tokeniser = PreTrainedTokenizerFast(
            tokenizer_file=tokeniser_path,
            special_tokens=special_tokens,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
            )

    dataset = load_data(infile_path)
    print("\nDATASET BEFORE SPLIT:\n", dataset)

    dataset = split_datasets(
        dataset["train"], train=split_train, test=split_test, val=split_val
        )
    print("\nDATASET AFTER SPLIT:\n", dataset)

    config = DistilBertConfig()
    model = DistilBertModel(config)

    print("\nSAMPLE DATASET ENTRY:\n", dataset["train"][0], "\n")

    col_torch = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    print(dataset)
    dataset.set_format(type='torch', columns=col_torch)
    dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=1)
    print("\nSAMPLE PYTORCH FORMATTED ENTRY:\n", next(iter(dataloader)))

    from transformers import DataCollatorForLanguageModeling
    from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokeniser),
        n_ctx=32,
        bos_token_id=tokeniser.bos_token_id,
        eos_token_id=tokeniser.eos_token_id,
    )
    config = DistilBertConfig()
    model = DistilBertModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"\nDistilBert size: {model_size/1000**2:.1f}M parameters")

    tokeniser.pad_token = tokeniser.eos_token
    data_collator = DataCollatorForLanguageModeling(tokeniser, mlm=False)
    out = data_collator([dataset["train"][i] for i in range(5)])
    for key in out:
        print(f"{key} shape: {out[key].shape}")

    args = TrainingArguments(
        output_dir="k12_cls_tmp",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        # fp16=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokeniser,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
    )
    os.environ["WANDB_DISABLED"] = "true"
    print(trainer)
    # trainer.train()
    # TODO: fix label issue
    # https://github.com/huggingface/transformers/issues/12631
    # https://stackoverflow.com/questions/58454157/pytorch-bert-typeerror-forward-got-an-unexpected-keyword-argument-labels
    # https://huggingface.co/course/chapter7/6?fw=pt
    # model.train()
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    # for epoch in range(3):
    #     for i, batch in enumerate(tqdm(data_collator)):
    #         batch = {k: v for k, v in batch.items()}
    #         outputs = model(**batch)
    #         loss = outputs[0]
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         if i % 10 == 0:
    #             print(f"loss: {loss}")

if __name__ == "__main__":
    main()
