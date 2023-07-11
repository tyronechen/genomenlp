#!/usr/bin/python
# make sure utils can be imported
import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(inspect.getfile(lambda: None)))

# get feature scores which weight a feature towards a class
import argparse
import contextlib
from hashlib import md5
from time import time
from warnings import warn
import screed
import wandb
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
from utils import html_to_pdf, _cite_me

def main():
    parser = argparse.ArgumentParser(
         description='Take complete classifier and calculate feature attributions.'
        )
    parser.add_argument('model_path', type=str,
                        help='path to local model directory OR wandb run')
    parser.add_argument('input_seqs', type=str, nargs="+",
                        help='input sequence(s) directly and/or fasta files')    
    parser.add_argument('-t', '--tokeniser_path', type=str, default="",
                        help='path to tokeniser.json file to load data from')
    parser.add_argument('-o', '--output_dir', type=str, default="./interpret_out",
                        help='specify path for output (DEFAULT: ./interpret_out)')
    parser.add_argument('-l', '--label_names', type=str, default=None, nargs="+",
                        help='provide label names matching order (DEFAULT: None).')
    parser.add_argument('-p', '--pdf_options', type=str, default=None,
                        help='load json of pdf conversion options (DEFAULT: \
                        {dpi: 300, page-size: A6, orientation: landscape})')

    args = parser.parse_args()
    model_path = args.model_path
    input_seqs = args.input_seqs
    tokeniser_path = args.tokeniser_path
    label_names = args.label_names
    output_dir = args.output_dir
    pdf_options = args.pdf_options
    if pdf_options is None:
        pdf_options = {'dpi': 300, 'page-size': 'A6', 'orientation': 'landscape'}

    print("\n\nARGUMENTS:\n", args, "\n\n")

    if os.path.exists(output_dir):
        warn(" ".join(["Overwriting files in:", output_dir]))
    else:
        os.mkdir(output_dir)

    if os.path.exists(model_path):
        print("Use model files from:", model_path)
    else:
        print("Downloading model files from wandb:", model_path)
        api = wandb.Api(timeout=10000)
        run = api.run(path=model_path)
        for i in tqdm(run.files(), desc="Downloading model files from wandb"):
            i.download(root=output_dir, replace=True) 
        model_path = output_dir
    
    if os.path.exists(tokeniser_path):
        print("Using custom tokeniser from:", tokeniser_path)
    else:
        print("Use tokeniser from:", model_path)
        tokeniser_path = model_path
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokeniser = AutoTokenizer.from_pretrained(tokeniser_path)

    cls_explainer = SequenceClassificationExplainer(
        model, tokeniser, custom_labels=label_names
        )

    for i in tqdm(input_seqs, desc="Calculating class attributions"):
        if os.path.exists(i):
            with screed.open(i) as infile:
                for j in infile:
                    cls_explainer(j.sequence.upper())
                    with contextlib.redirect_stdout(None): 
                        outfile_path = "".join([output_dir, "/", j.name, ".html"])
                        cls_explainer.visualize(outfile_path)
                        with open(outfile_path, 'r') as tmp:
                            data = tmp.read()    
                        with open(outfile_path, 'w') as tmp:
                            tmp.write(data.replace('▁', ''))                        
                        html_to_pdf(outfile_path, options=pdf_options)
        else:
            cls_explainer(i)
            unique = md5(str(time()).encode("utf-8")).hexdigest()[:16]
            with contextlib.redirect_stdout(None):
                outfile_path = "".join([output_dir, "/", i[:16], "_", unique, ".html"])
                cls_explainer.visualize(outfile_path)
                with open(outfile_path, 'r') as tmp:
                    data = tmp.read()    
                with open(outfile_path, 'w') as tmp:
                    tmp.write(data.replace('▁', '')) 
                html_to_pdf(outfile_path, options=pdf_options)

if __name__ == "__main__":
    main()
    _cite_me()