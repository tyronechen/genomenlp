# Expected
1. TFIDF(sequence_file, k_low, k_high,label) --- expect output as list of [X,Y], X = array of vectors of eacg sequence in sequence file, Y = label of sequence_file processed --- X and Y will have same first dimension
2. tfidf_n_topfeatures(sequence_file, k_low, k_high,n) --- expect n top features of tfidfvectriser trained in form of list
3. visualise_cvec_freq_dist(sequence_file, k_low, k_high) --- frequency distribution of top 50 tokens from training tfidfvectoriser 

# Obtained
Same as expected
All three functions mentioned above works successfully

# Arg
sequence_file = data file we want to process
k_low, k_high = range of k_mers to form a sentence from dna dataset
n = no of top features we want
label = label your data trained for classification downstream - 1 for natural sequence, 0 for synthetic sequence

# IO
input file is 'PromoterSet.fa' which can be found on below link:
https://drive.google.com/drive/folders/1WBh1ek_-i46sU412ycWtMSdA26LPnFxK?usp=sharing

output is decribed in expected for each of functions
​
# Environment
sklearn = 1.0.2
TfidfVectorizer
FreqDistVisualizer
