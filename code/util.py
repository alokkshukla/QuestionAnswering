from qa_data import PAD_ID
import numpy as np
import random
import linecache

def get_one_hot(idx, max_size= 750):
	one_hot_vec = [0 for x in range(max_size)]
	one_hot_vec[int(idx)] = 1
	return np.array(one_hot_vec)

def pad_sequences(question, paragraph, max_q = 750, max_p = 750):
	'''
	Function to pad sequences
	'''
	if(len(paragraph) > max_p):
		paragraph = paragraph[:max_p]
	else:
		for i in range(max_p - len(paragraph)):
			paragraph.append(PAD_ID)
	for i in range(max_q - len(question)):
		question.append(PAD_ID)
	return (question, paragraph)

def mask_sequences(question, paragraph):
	'''
	A function to mask question and paragraph at once
	'''
	q_mask = []
	for i in question:
		if i == PAD_ID:
			q_mask.append(0)
		else:
			q_mask.append(1)
	p_mask = []
	for i in paragraph:
		if i == PAD_ID:
			p_mask.append(0)
		else:
			p_mask.append(1)
	return (q_mask, p_mask)

def load_validate(file1, file2, file3, sample = 100):
	'''
	Function to load validation dataset.
	This loads a random number of samples. Mainly used to quickly check how the system is doing.
	'''
	selection_list = range(4284)
	random_samples = random.sample(selection_list, sample)
	batch = []
	for i in random_samples:
		line1, line2, line3 = linecache.getline(file1, i), linecache.getline(file2, i), linecache.getline(file3, i)
		question, paragraph = pad_sequences(line1.strip().split(), line2.strip().split())
		q_mask, p_mask = mask_sequences(question, paragraph)
		a_s = get_one_hot(line3.strip().split()[0])
		a_e = get_one_hot(line3.strip().split()[1])
		batch.append((question, paragraph, a_s, a_e, q_mask, p_mask))
	return batch

def load_dataset(file1, file2, file3, batch_size, size_of_dataset = 81381):
	'''
	Function to load training dataset
	'''
	f_1 = open(file1,'r')
	f_2 = open(file2,'r')
	f_3 = open(file3,'r')
	batch = []
	for i in range(size_of_dataset):
		line1 = f_1.readline()
		line2 = f_2.readline()
		line3 = f_3.readline()
		question, paragraph = pad_sequences(line1.strip().split(), line2.strip().split())
		q_mask, p_mask = mask_sequences(question, paragraph)
		a_s = get_one_hot(line3.strip().split()[0])
		a_e = get_one_hot(line3.strip().split()[1])
		batch.append((question, paragraph, a_s, a_e, q_mask, p_mask))
		if len(batch) == batch_size:
			yield batch
			batch = []
		elif i == size_of_dataset - 1:
			yield batch
			batch = []
