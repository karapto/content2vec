# import dependencies
%matplotlib inline
import pandas as pd
import numpy as np
import multiprocessing
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
from collections import Counter
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns
import gensim.models.word2vec as w2v
import sklearn.manifold
import time
from process import sent_tokenizer
from process import sentence_cleaner
from process import apply_all
from process import doc_length
from process import nearest_similarity_cosmul
sns.set_style("darkgrid")

def main():
	df = pd.read_csv('../input/fake.csv', usecols = ['uuid','author','title','text','language','site_url','country'])
	df = df[df.language == 'english']
	df['title'].fillna(value="", inplace=True)
	df.dropna(axis=0, inplace=True, subset=['text'])
	df = df.sample(frac=1.0) # shuffle the data
	df.reset_index(drop=True,inplace=True)

	df['sent_tokenized_text'] = df['text'].apply(apply_all)
	print("Time to clean and tokenize", len(df), "articles:", (t2-t1)/60, "min")

	all_words = [word for item in list(df['sent_tokenized_text']) for word in item]
all_words = [subitem for item in all_words for subitem in item]

	fdist = FreqDist(all_words)
	len(fdist) # number of unique words
	fdist.most_common(20)
	
	# choose k and visually inspect the bottom 10 words of the top k
	k = 50000
	top_k_words = fdist.most_common(k)
	top_k_words[-10:]
	
	# choose k and visually inspect the bottom 10 words of the top k
	k = 30000
	top_k_words = fdist.most_common(k)
	top_k_words[-10:]

	# document length
	df['doc_len'] = df['sent_tokenized_text'].apply(doc_length)
	doc_lengths = list(df['doc_len'])
	df.drop(labels='doc_len', axis=1, inplace=True)

	print("length of list:",len(doc_lengths),
      "\naverage document length", np.average(doc_lengths),
      "\nmaximum document length", max(doc_lengths))
	
	all_sentences = list(df['sent_tokenized_text'])
	all_sentences = [subitem for item in all_sentences for subitem in item]
	all_sentences[:2] # print first 5 sentences

	token_count = sum([len(sentence) for sentence in all_sentences])
	print("The corpus contains {0:,} tokens".format(token_count)) # total words in corpus

	num_features = 300 # number of dimensions
	# if any words appear less than min_word_count amount of times, disregard it
	# recall we saw that the bottom 10 of the top 30,000 words appear only 7 times in the corpus, so lets choose 10 here
	min_word_count = 10
	num_workers = multiprocessing.cpu_count()
	context_size = 7 # window size around target word to analyse
	downsampling = 1e-3 # downsample frequent words
	seed = 1 # seed for RNG
	
	# setting up model with parameters above
	content2vec = w2v.Word2Vec(
    		sg=1,
		seed=seed,
    		workers=num_workers,
    		size=num_features,
   		min_count=min_word_count,
    		window=context_size,
    		sample=downsampling
	)

	content2vec.build_vocab(all_sentences)
	print("Word2Vec vocabulary length:", len(content2vec.wv.vocab))
	content2vec.corpus_count
	content2vec.train(all_sentences, total_examples=content2vec.corpus_count, epochs=content2vec.iter)
	all_word_vectors_matrix = content2vec.wv.syn0
	all_word_vectors_matrix.shape # .shape[0] are the top words we are considering in training word2vec
	# train tsne model for visualisation
	tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
	all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
	print("time to train TSNE on", all_word_vectors_matrix.shape[0], "word vectors:", (t2-t1)/60, "min")
	
	# create a dataframe *points* to store the 2D embeddings of all words
	points = pd.DataFrame(
    	[
        	(word, coords[0], coords[1])
        	for word, coords in [
            		(word, all_word_vectors_matrix_2d[content2vec.wv.vocab[word].index])
            		for word in content2vec.wv.vocab
       		 ]
    	],
    	columns=["word", "x", "y"]
	)

	content2vec.most_similar('trump')
	nearest_similarity_cosmul("trump", "presidentelect", "clinton") # makes sense

if __name__ == "__main__":
    main() 
