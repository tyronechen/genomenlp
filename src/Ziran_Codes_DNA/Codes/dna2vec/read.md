# Expected
Install dna2vec package
Install requirements
Refer Google Drive link: https://drive.google.com/drive/folders/1WBh1ek_-i46sU412ycWtMSdA26LPnFxK?usp=sharing
Refere Google Colab link:
    https://colab.research.google.com/drive/1X4zwWBqqzV1zu7HYuD1t6hH8BTpupDqY?usp=sharing
1. rename_folder(in_folder, dst_folder): converts name of file and location of input files to required format as required by dna2vec. dna2vec needs files in dna2vec/inputs/hg38 and files names should chr{int}, then only successful training will happen.
2. main(): dna2vec training function to generate embeddings. Combines multiple files, functions from dna2vec package.

# Obtained
1. rename_folder(in_folder, dst_folder): converts any file location and filename to required format by dna2vec. Successful
2. main() is not able to run----attic_util error----may run successfully in command line

# Arg
1. rename_folder(in_folder, dst_folder):
in_folder = folder containing inputs initially
dst_foder = folder where you want your inputs to go.

# IO
input file is 'PromoterSet.fa' which can be found on below link:
https://drive.google.com/drive/folders/1WBh1ek_-i46sU412ycWtMSdA26LPnFxK?usp=sharing


​
# Environment
requirements.txt of dna2vec package uploaded at https://drive.google.com/drive/folders/1WBh1ek_-i46sU412ycWtMSdA26LPnFxK?usp=sharing
os
shutil
