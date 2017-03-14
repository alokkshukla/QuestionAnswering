from qa_data import PAD_ID
import random
import linecache

def pad_sequences (q, p, q_max = 750, p_max = 750):
	
	if(len(p) > p_max):
		p = p[:p_max]
	else:
		for i in range(p_max - len(p)):
			p.append(PAD_ID)

	for i in range(q_max - len(q)):
		q.append(PAD_ID)

	return (q,p)

def mask_sequences(q,p):

	q_mask = []
	for i in q:
		if i == PAD_ID:
			q_mask.append(0)
		else:
			q_mask.append(1)
	p_mask = []
	for i in q:
		if i == PAD_ID:
			p_mask.append(0)
		else:
			p_mask.append(1)

	return (q_mask, p_mask)

def get_one_hot(idx, max_size= 750):

	one_hot_vec = [0 for x in range(max_size)]
	one_hot_vec[int(idx)] = 1

	return one_hot_vec

def load_dataset(f1, f2, f3, batch_size, size_of_dataset = 81382):
	
	fd1 = open(f1,'r')
	fd2 = open(f2,'r')
	fd3 = open(f3,'r')

	batch = []
	
	for i in range(size_of_dataset):

		line1 = fd1.readline()
		line2 = fd2.readline()
		line3 = fd3.readline()

		q, p = pad_sequences(line1.strip().split(),line2.strip().split())
		q_mask, p_mask = mask_sequences(q, p)
		a_s = get_one_hot(line3.strip().split()[0])
		a_e = get_one_hot(line3.strip().split()[1])
		batch.append((q, p, a_s, a_e, q_mask, p_mask))

		if len(batch) == batch_size:
			yield batch
			batch = []

def load_validate(f1, f2, f3, sample):
	size_of_dataset = 4285
	sel_list = range(size_of_dataset)
	ran_100 = random.sample(sel_list, sample)

	batch = []
	for i in ran_100:
		line1, line2, line3 = linecache.getline(f1, i), linecache.getline(f2, i),linecache.getline(f3, i)
		q, p = pad_sequences(line1.strip().split(),line2.strip().split())
		q_mask, p_mask = mask_sequences(q, p)
		a_s = get_one_hot(line3.strip().split()[0])
		a_e = get_one_hot(line3.strip().split()[1])
		batch.append((q, p, a_s, a_e, q_mask, p_mask))

	return batch
