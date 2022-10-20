# Expected
Sentence similar to english sentences are generated for each sequence in sequence_file
​
# Obtained
Same as expected
Kmers_funct, spacings, loadsequence works successfully

# Arg
sequence_file = data file we want to process
k_low, k_high = range of k_mers to form a sentence from dna dataset

# IO
input file is 'PromoterSet.fa' which can be found on below link:
https://drive.google.com/drive/folders/1WBh1ek_-i46sU412ycWtMSdA26LPnFxK?usp=sharing

output will be corpus of sentences in form of list to be used for cvec, tfidf downstream
each component of list represents sentence generated from each sequence in file
​
# Environment
fastaparser
random
numpy=1.21.6
pandas=1.3.5
