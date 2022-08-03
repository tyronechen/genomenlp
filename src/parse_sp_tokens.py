import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast

def main():
    parser = argparse.ArgumentParser(
         description='Parse SentencePiece output json file into a python object\
          usable by other modules'
        )
    parser.add_argument('tokeniser_path', type=str,
                        help='path to tokeniser.json file to load data from')
    parser.add_argument('-o', '--outfile_dir', type=str, default="./",
                        help='path to output plot directory')

    args = parser.parse_args()
    tokeniser_path = args.tokeniser_path

    if not os.path.exists(tokeniser_path):
        raise OSError("File does not exist!")

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
    tokens = list(tokeniser.vocab.keys())
    tokens = [x.replace("▁", "") for x in tokens]
    for special_token in special_tokens:
        tokens.remove(special_token)
    tokens_len = [len(x) for x in tokens]

    for_plot = pd.DataFrame(pd.Series(tokens_len))
    print(for_plot)
    for_plot.columns = ["Selected k-mer lengths (base pairs)"]
    for_plot.index.name = "Quantity (units)"

    hist = for_plot.plot(kind="hist", grid=False, legend=False)
    hist.set_xlabel("Selected k-mer lengths (base pairs)")
    title = "".join(
        ["Selected k-mer length distribution (of ", str(len(tokens_len)), ")"]
        )
    hist.set_title(title)
    plt_out = ["".join(
        [outfile_dir, "kmer_length_histogram.", i]
        ) for i in ["pdf", "png"]]
    [plt.savefig(i, dpi=300) for i in plt_out]

if __name__ == "__main__":
    main()
