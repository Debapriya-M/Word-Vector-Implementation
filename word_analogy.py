import os
import pickle
import numpy as np
from scipy import spatial
import sys

model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

sample_text = open('word_analogy_test.txt', 'r')
#f = open("word_analogy_test_predictions_cross_entropy.txt", "w")
f = open("word_analogy_test_predictions_nce.txt", "w")
for line in sample_text:
	text = line.split('||')
	print(text)
	left_text = text[0]
	right_text = text[1]
	print('Left Text: ', left_text)
	print('Right Text: ', right_text)
	left_word_array = left_text.split(',')
	print('left_word_array: ', left_word_array)
	words = list()
	word = list()
	for x in left_word_array:
		a = x.replace('"','')
		b = a.replace('\n',' ')
		words.append(a.split(':'))
		print('Words --->', words)
		word_diff = 0
		sum = 0
		for i in words:
			word1 = i
			w1 = word1[0]
			print(w1)
			w2 = word1[1]
			print(w2)
			word_diff = embeddings[dictionary[w1]] - embeddings[dictionary[w2]]
			sum = sum + word_diff
		avg = sum/3
		print('Dimension of Average --->',avg.shape)

	right_word_array = right_text.split(',')
	print('right_word_array ', right_word_array)
	rightwords = list()
	minVal = -sys.maxsize -1
	maxVal = sys.maxsize
	ar = right_text.replace('"','')
	print('ar', ar)
	br = ar.replace('\n','')
	print ('br', br)
	index_min = 0
	index_max = 0
	cosineSimilarity = list()
	rightwords.append(br.split(','))
	print(rightwords)
	for rw in rightwords:
		index = 0
		for r in rw:
			r0 = r.split(':')[0]
			print('r0', r0)
			r1 = r.split(':')[1]
			right_word_diff = embeddings[dictionary[r0]] - embeddings[dictionary[r1]]

			cosineSimilarity.append(abs(1 - spatial.distance.cosine(right_word_diff, avg)))
			
		minIndex = cosineSimilarity.index(min(cosineSimilarity))
		maxIndex = cosineSimilarity.index(max(cosineSimilarity))
		print(minIndex)
		print(minIndex)

		f.write(right_text.replace(',', ' ').replace('\n', '') + " " + right_word_array[minIndex].replace('\n','') + " " + right_word_array[maxIndex].replace('\n','') + "\n")
	rightwords = list()

	
f.close()















