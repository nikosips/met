import sys
import math

import faiss
import numpy as np
import torch
import torch.nn as nn

from code.utils.utils import *
from code.classifiers.knn_classifier import *




def train_contrastive_1epoch_virtual(model,criterion,optimizer,train_loader,epoch,vbsizemul):
	'''Train model with the contrastive loss for one-epoch. 
	Supports virtual batch training with gradient accumulation
	in order to have larger batches with limited hardware.
	'''

	# set model to train mode
	model.train()

	model.backbone.apply(set_batchnorm_eval) #don't change the statistics of the BN layers learned on ImageNet

	epoch_loss = 0.0
	batch = 1
	
	for i,(pair,targets) in enumerate(train_loader):

		targets = targets.cuda()

		#forward pass
		embeds = model(pair[0].cuda(),pair[1].cuda())

		#loss calculation
		loss = criterion(embeds[0],embeds[1],targets)/vbsizemul

		#backward_pass
		loss.backward()

		#gradient descent
		if (i+1) % vbsizemul == 0 or (i+1) == len(train_loader):

			optimizer.step()
			optimizer.zero_grad()

			# print statistics
			progress(loss=loss.data.item(),
					 epoch=epoch,
					 batch=batch,
					 batch_size=vbsizemul*train_loader.batch_size,
					 dataset_size=len(train_loader.dataset))

			batch +=1

		epoch_loss += ((loss.item()*pair[0].size(0))/len(train_loader.dataset))*vbsizemul

	print("epoch : " + str(epoch) + " finished, train loss = " + str(epoch_loss))

	return epoch_loss



def save_checkpoint(state,filename,epoch):

	torch.save(state,filename + "_epoch:_" + str(epoch))



def set_batchnorm_eval(m):
	'''Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
	'''

	classname = m.__class__.__name__
	if classname.find('BatchNorm') != -1:
		# freeze running mean and std:
		# we do training one image at a time
		# so the statistics would not be per batch
		# hence we choose freezing (ie using imagenet statistics)
		m.eval()
		# # freeze parameters:
		# # in fact no need to freeze scale and bias
		# # they can be learned
		# # that is why next two lines are commented
		# for p in m.parameters():
			# p.requires_grad = False



def progress(loss, epoch, batch, batch_size, dataset_size):

	batches = math.ceil(float(dataset_size) / batch_size)
	count = batch * batch_size
	bar_len = 40
	filled_len = int(round(bar_len * count / float(dataset_size)))

	bar = '=' * filled_len + '-' * (bar_len - filled_len)

	status = 'Epoch {}, Batch Loss: {:.8f}'.format(epoch, loss)
	_progress_str = "\r \r [{}] ...{}".format(bar, status)
	sys.stdout.write(_progress_str)
	sys.stdout.flush()

	if batch == batches:
		print()



def validate(net,train_loader,train_labels,val_loader,val_labels,ret_train_descr = False,train_descr = None):
	
	print("Validation phase")

	#descriptor extraction (singlescale for validation)
	if train_descr is None:
		train_descr = extract_embeddings(net,train_loader,ms = [1],msp = 1.0)

	val_descr = extract_embeddings(net,val_loader,ms = [1],msp = 1.0)

	train_descr = np.ascontiguousarray(train_descr,dtype=np.float32)
	val_descr = np.ascontiguousarray(val_descr,dtype=np.float32)

		
	clf = KNN_Classifier(K = 1,t = 1)
	clf.fit(train_descr,train_labels)

	val_preds,val_confs = clf.predict(val_descr)
	val_gap,val_non_distr_gap,val_acc = evaluate(np.array(val_preds),np.array(val_confs),val_labels)


	if ret_train_descr:
		return val_gap,val_non_distr_gap,val_acc,train_descr

	else:
		return val_gap,val_non_distr_gap,val_acc



def mine_negatives(image_paths,root,image_descrs,image_labels):

	image_descrs = np.ascontiguousarray(image_descrs,dtype=np.float32)
	
	res = faiss.StandardGpuResources()

	index = faiss.IndexFlatIP(np.shape(image_descrs)[1])
	index = faiss.index_cpu_to_gpu(res, 0, index)
	
	index.add(image_descrs.astype("float32"))

	#find top 25 neighbors
	#they might contain images from the same class as the query, not all are negatives yet
	similarities,idxs = index.search(image_descrs.astype("float32"),25)

	negs_all = []

	#clean those neighbors from the candidate positives 
	#form a list for each sample with its 10 closest negatives

	for j,idx in enumerate(idxs):

		negs_one = []

		i = 0
		while len(negs_one)<10:

			if image_labels[idx[i]] != image_labels[j]:
				negs_one.append(image_paths[idx[i]])

			i+=1

		#pick one of the 10 closest neighbors
		indices = np.arange(10)
		np.random.shuffle(indices)
		index = indices[0]

		negs_all.append(negs_one[index])

	return negs_all



def create_class_idx_dict(targets):
	'''Create a dict that stores the index of the samples of every class.
	'''

	class_idx_dict = {}

	unique_classes = np.unique(targets)
	
	print("creating class idx dict")
	for i,class_id in enumerate(unique_classes):
		idxs = np.where(targets == class_id)[0]
		class_idx_dict[class_id] = idxs
	print("class idx dict done")

	return class_idx_dict



def mine_positive(i,same_class_sample_idxs,train_descr):

	similarities = np.dot(train_descr[same_class_sample_idxs],train_descr[i])

	#take two top(one is itself) and pick randomly one of two
	return same_class_sample_idxs[np.random.choice(np.argsort(-similarities)[:2],1)[0]]