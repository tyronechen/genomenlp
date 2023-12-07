#!/usr/bin/python
# download human genome, extract primary assembly, tokenise
# wget -O 'GRCh38.p14.zip' 'https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCF_000001405.40/download?include_annotation_type=GENOME_FASTA,GENOME_GFF,RNA_FASTA,CDS_FASTA,PROT_FASTA,SEQUENCE_REPORT'
# unzip 'GRCh38.p14.zip'
# ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna
import os
import screed
from tqdm import tqdm

def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

def main():
    infile_path = "ncbi_dataset/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna"
    outfile_dir = "out_segments"
    os.makedirs(outfile_dir, exist_ok=True)
    db = screed.read_fasta_sequences(infile_path)
    keep = [
        "NC_000001.11",
        "NC_000002.12",
        "NC_000003.12",
        "NC_000004.12",
        "NC_000005.10",
        "NC_000006.12",
        "NC_000007.14",
        "NC_000008.11",
        "NC_000009.12",
        "NC_000010.11",
        "NC_000011.10",
        "NC_000012.12",
        "NC_000013.11",
        "NC_000014.9",
        "NC_000015.10",
        "NC_000016.10",
        "NC_000017.11",
        "NC_000018.10",
        "NC_000019.10",
        "NC_000020.11",
        "NC_000021.9",
        "NC_000022.11",
        "NC_000023.11",
        "NC_000024.10",
    ]
    seqs_len = 2**9
    seqs_per_file = 2**16

    for chr in keep:
        seq = db[chr]
        print("Chr len:", seq.name, len(seq))
        count_entry = 0
        count_files = 1
        for chunk in tqdm(chunkstring(str(seq.sequence).upper(), seqs_len), desc="Segmenting chr"):
            count_entry += 1
            fname = "".join([outfile_dir, "/", seq.name, "_", str(count_files), ".fasta"])
            if count_entry == seqs_per_file:
                count_files += 1
                fname = "".join([outfile_dir, "/", seq.name, "_", str(count_files), ".fasta"])
                count_entry = 0
            else:
                entry = "".join([">", seq.name, " ", str(count_entry), "\n", chunk, "\n"])
                with open(fname, mode="a+") as out:
                    out.write(entry)


if __name__ == "__main__":
    main()