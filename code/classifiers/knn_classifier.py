import faiss
import itertools
import numpy as np

from code.utils.utils import *




class KNN_Classifier():


	def __init__(self,K,t):

		self.K = K #number of neighbors to take into account
		self.t = t #temperature of the softmax


	def fit(self, train_descrs, train_labels):

		res = faiss.StandardGpuResources()  # use a single GPU

		self.index = faiss.IndexFlatIP(np.shape(train_descrs)[1])
		self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
		self.index.add(train_descrs)

		self.train_labels = train_labels
		self.n = np.unique(train_labels).shape[0] #total number of classes in the train set


	def predict(self, test_descrs):

		similarities_k_sorted, idx = self.index.search(test_descrs,self.K)
		train_labels_k_sorted = self.train_labels[idx]

		preds,confs = [],[]
		
		for i in range(np.shape(similarities_k_sorted)[0]):

			unique_neighbors = np.unique(train_labels_k_sorted[i])
			count_neighbors = np.zeros((1,len(unique_neighbors)))[0]
			total_sims = np.zeros((1,len(unique_neighbors)))[0]

			for j in range(len(train_labels_k_sorted[i])):

				idx_total = np.where(unique_neighbors==train_labels_k_sorted[i][j])[0]
				
				if len(idx_total)==0:
					continue
				
				total_sims[idx_total] = max(total_sims[idx_total], similarities_k_sorted[i][j])

				
			total_sims = np.exp(self.t*total_sims)
			total_sims /= (total_sims.sum()+(self.n-total_sims.shape[0])*np.exp(0))

			test_label_pred = unique_neighbors[total_sims.argmax()]
			confidence = total_sims.max()

			preds.append(test_label_pred)
			confs.append(confidence)

		return preds,confs



def tune_KNN(param_grid,train_descr,train_labels,val_descr,val_labels,verbose = True):
	'''Tuning is performed on the GAP metric
	'''
	combinations = itertools.product(*(param_grid[p] for p in param_grid))
	best_score = -np.Inf
	best_params = {}

	for param_set in combinations:

		clf = KNN_Classifier(K=int(param_set[0]),t=float(param_set[1]))
		clf.fit(train_descr,train_labels)
		val_preds,val_confs = clf.predict(val_descr)
		print(param_set)
		gap,_,_ = evaluate(val_preds,val_confs,val_labels,verbose=verbose)

		score = gap
		if score > best_score:
			best_score = score
			best_params = param_set

	return best_score,dict(zip(param_grid, best_params))
