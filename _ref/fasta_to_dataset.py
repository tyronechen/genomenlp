import argparse
import gzip
import itertools
import os
import sys
from math import ceil
from random import choices, shuffle
from warnings import warn
from datasets import ClassLabel, Dataset, DatasetDict, Value
import pandas as pd
import screed
import torch
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast

def bootstrap_seq(seq: str, block_size: int=2):
    """Take a string and shuffle it in blocks of N length"""
    chunks, chunk_size = len(seq), block_size
    seq = [ seq[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    shuffle(seq)
    return "".join(seq)

def generate_from_freq(seq: str, block_size: int=2,
                       alphabet: list=["A","C","G","T"], offset: float=0.01):
    """Take a string and sample from freq distribution to fill up seq length"""
    if len(seq) == 0:
        return
    assert block_size <= len(seq), "Block size exceeds sequence length!"
    to_count = ["".join(i) for i in itertools.product(alphabet, repeat=block_size)]
    count = [str.count(seq, x) + offset for x in to_count]
    freq = dict(zip(to_count, [x / len(count) for x in count]))
    draw_size = ceil(len(seq) / block_size)
    new = choices(list(freq.keys()), weights=freq.values(), k=draw_size)
    return "".join(new)[:len(seq)]

def reverse_complement(dna: str):
    """Take a dna string as input and return reverse complement"""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return "".join([complement[base] for base in dna[::-1]])

def dataset_to_disk(dataset: Dataset, outfile: str):
    """Take a 🤗 dataset object, path as output and write files to disk"""
    if os.path.exists(outfile):
        warn("".join(["Overwriting contents of file!: ", outfile]))
    if outfile.endswith(".csv"):
        dataset.to_csv(outfile)
    elif outfile.endswith(".json"):
        dataset.to_json(outfile)
    elif outfile.endswith(".parquet"):
        dataset.to_parquet(outfile)
    elif outfile != "":
        dataset.save_to_disk(outfile)

def main():
    parser = argparse.ArgumentParser(
        description='Take fasta-like files, convert to HuggingFace🤗  dataset. \
        Accepts .gz files.'
    )
    parser.add_argument('infile_path', type=str, help='path to fasta/gz file')
    parser.add_argument('-b', '--block_size', type=int, default=2,
                        help='size of block to permute real seqs into \
                        synthetic null seqs as negative control (DEFAULT: 2)')
    parser.add_argument('-c', '--control_dist', type=str, default="frequency",
                        help='generate control distribution by [ bootstrap | \
                        frequency | /path/to/file ] (DEFAULT: frequency)')
    parser.add_argument('-o', '--outfile', type=str, default="hf_out/",
                        help='write 🤗 dataset to disk as \
                        [ csv | json | parquet | dir/ ] (DEFAULT: "hf_out/")')
    parser.add_argument('-t', '--tokeniser_path', type=str, default="",
                        help='path to tokeniser.json file to save or load data')
    parser.add_argument('--no_reverse_complement', action="store_false",
                        help='turn off reverse complement, has no effect if \
                        loading an existing tokeniser (DEFAULT: True)')

    args = parser.parse_args()
    infile_path = args.infile_path
    block_size = args.block_size
    control_dist = args.control_dist
    outfile = args.outfile
    tokeniser_path = args.tokeniser_path
    do_reverse_complement = args.no_reverse_complement

    i = " ".join([i for i in sys.argv[0:]])
    print("COMMAND LINE ARGUMENTS FOR REPRODUCIBILITY:\n\n\t", i, "\n")

    # do reverse complement and negative control generation in the same loop
    seqs = dict()
    null = dict()

    if control_dist == "bootstrap" or control_dist == "frequency":
        if os.path.isfile(control_dist):
            raise OSError("Why are you naming your control distribution file \
                          the same as a command line argument? (╯°□°)╯︵ ┻━┻")
    else:
        with screed.open(control_dist) as nullfile:
            for read in nullfile:
                head = read.name
                seq = read.sequence.upper()
                null["_".join(["NULL", head])] = seq
                if do_reverse_complement is True:
                    rc = reverse_complement(seq)
                    null["_".join(["NULL", head, "RC"])] = rc

    with screed.open(infile_path) as seqfile:
        for read in seqfile:
            head = read.name
            seq = read.sequence.upper()
            seqs[head] = seq

            if do_reverse_complement is True:
                rc = reverse_complement(seq)
                seqs["_".join([head, "RC"])] = rc

            if control_dist == "bootstrap":
                null["_".join(["NULL", head])] = bootstrap_seq(seq, block_size)
                if do_reverse_complement is True:
                    null["_".join(["NULL", head, "RC"])] = bootstrap_seq(rc, 2)
            if control_dist == "frequency":
                null["_".join(["NULL", head])] = generate_from_freq(seq, block_size)
                if do_reverse_complement is True:
                    null["_".join(["NULL", head, "RC"])] = generate_from_freq(rc, 2)

    # TODO: can add more seq metadata into cols if needed
    seqs = pd.DataFrame({"feature": seqs}).reset_index()
    seqs.rename(columns={"index": "idx"}, inplace=True)
    seqs["labels"] = 1

    null = pd.DataFrame({"feature": null}).reset_index()
    null.rename(columns={"index": "idx"}, inplace=True)
    null["labels"] = 0

    # configure data into a huggingface compatible dataset object
    # see https://huggingface.co/docs/datasets/access
    data = pd.concat([seqs, null], axis=0).sort_index().reset_index(drop=True)
    data = Dataset.from_pandas(data)

    # we can tokenise separately if needed from the positive data
    # see https://huggingface.co/docs/transformers/fast_tokenizers for ref
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    if os.path.exists(tokeniser_path):
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
    else:
        print("WRITING NEW TOKENISER:", tokeniser_path)
        tokeniser = SentencePieceUnigramTokenizer()
        tokeniser.train_from_iterator(
            seqs["feature"],
            unk_token="<unk>",
            vocab_size=2000,
            show_progress=True,
            special_tokens=special_tokens,
        )
        tokeniser.save(tokeniser_path)
        tokeniser = PreTrainedTokenizerFast(
            tokenizer_object=tokeniser,
            special_tokens=special_tokens,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
            )

    # configure dataset object to correctly assign class labels
    # see https://huggingface.co/docs/datasets/process for reference
    print("\nASSIGNING CLASS LABELS:")
    # TODO: add padding
    # https://discuss.huggingface.co/t/padding-in-datasets/2345/5
    dataset = data.map(lambda data: tokeniser(data['feature']), batched=True)
    tmp = dataset.features.copy()
    tmp["labels"] = Value('bool')
    tmp["labels"] = ClassLabel(num_classes=2, names=['negative', 'positive'])
    tmp["idx"] = Value('string')
    dataset = dataset.cast(tmp)
    print("\nDATASET FEATURES:\n", dataset.features, "\n")
    print("CLASS NAMES:", dataset.features["labels"].names)
    print("CLASS COUNT:", dataset.features["labels"].num_classes)
    print("\nDATASET HAS FOLLOWING SPECIFICATIONS:\n", dataset)

    print("\nWRITING 🤗 DATA TO DISK (OVERWRITES ANY EXISTING!) AT:\n", outfile)
    dataset_to_disk(dataset, outfile)

    print("\nSAMPLE DATASET ENTRY:\n", dataset[0], "\n")

    print("SAMPLE TOKEN MAPPING FOR FIRST 5 TOKENS IN SEQ:",)
    for i in dataset[0]["input_ids"][0:5]:
        print("TOKEN ID:", i, "\t|", "TOKEN:", tokeniser.decode(i))

    col_torch = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    dataset.set_format(type='torch', columns=col_torch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    print("\nSAMPLE PYTORCH FORMATTED ENTRY:\n", next(iter(dataloader)))

if __name__ == "__main__":
    main()
