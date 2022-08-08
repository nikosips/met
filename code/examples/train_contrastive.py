import argparse
import os
import sys
import pickle
import math
import numpy as np

from code.utils.train_utils import *
from code.networks.SiameseNet import *
from code.utils.datasets import *
from code.utils.utils import *
from code.utils.losses import *
from code.utils.augmentations import augmentation


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None #used to suppress some warnings



'''Script for training a backbone with the contrastive loss on pairs of the Met dataset.
Each image in the dataset is used as anchor once and contributes to two pairs: one negative and one positive,
where in both it is used as anchor.
The possible types of pairs are the following:

- (synthetic positive) : corresponds to pairs_type == sim_siam_pos
- (synthetic positive,hard negative) : corresponds to pairs_type == sim_siam_pos+new_neg
- (synthetic + real positive,hard negative) : corresponds to pairs_type == pos+new_neg
- (synthetic + real closest positive,hard negative) : corresponds to pairs_type == new_pos+new_neg

'''


def main():

	parser = argparse.ArgumentParser()

	parser.add_argument('directory', metavar='EXPORT_DIR',help="directory where the trained model's checkpoints will be saved at")
	parser.add_argument('--gpuid', default=0, type=int)
	parser.add_argument('--net', default='resnet18')
	parser.add_argument('--mini', action='store_true') #choose whether to train on mini Met training set or not
	parser.add_argument('--backbone_lr', default=1e-7) #lr for the optimization algorithm (same for the entire encoder)
	parser.add_argument('--bsize', default=64) #batch size for the optimization algorithm
	parser.add_argument('--epochs', default=10) #number of epochs the model will be trained for (1 epoch: each image is used as anchor once)
	parser.add_argument('--vbsizemul', default=1, type=int)
	parser.add_argument('--initeval', action='store_true')
	parser.add_argument('--margin', default=1.8, type=float) #margin of the contrastive loss
	parser.add_argument('--wdecay', default=1e-6, type=float)
	parser.add_argument('--sched_step', default=6, type=int)
	parser.add_argument('--sched_gamma', default=0.1, type=float)
	parser.add_argument('--pretrained', action='store_true') #start from the result of Imagenet pretraining
	parser.add_argument('--imsize', default=500, type=int) #image size to train on
	parser.add_argument('--resume', default=None, type=str) #resume training from a given checkpoint
	parser.add_argument('--pairs_type', default='sim_siam_pos+new_neg', type=str) 
	parser.add_argument('--emb_proj', action='store_true') #if the backbone will also have an FC layer
	parser.add_argument('--pca', action='store_true') #additionaly initialize FC layer with pca
	parser.add_argument('--seed', default=None, type=int)
	parser.add_argument('--init_descr', type = str, default= None) #descriptors that are output by the backbone before training; should be singlescale
	parser.add_argument('--info_dir',default=None, type=str, help = 'directory where ground truth is stored') 
	parser.add_argument('--im_root',default=None, type=str, help = 'directory where images are stored')

	args = parser.parse_args()

	if args.seed is not None:
		
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)
		np.random.seed(args.seed)

	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)

	if not os.path.exists(args.directory):
		os.makedirs(args.directory, exist_ok=True)

	print("Will save trained model in {}".format(args.directory))
	print("Will train for "+str(args.epochs)+" epochs")
	print("Batch size used: " + str(int(args.bsize)*args.vbsizemul))

	checkpoint_path = ("method:_contrastive_"+"net:_"+args.net+"_bckbn_lr:"+str(args.backbone_lr)+
				 "_b_size:_"+str(int(args.bsize)*args.vbsizemul)+"_epochs:_"+str(args.epochs)+
				 "_wdecay:_" + str(args.wdecay)+ "_margin:_" + str(args.margin)+"_schedstep:_"+str(args.sched_step)+
				 "_schedgamma:_"+str(args.sched_gamma)+"_imsize:_"+str(args.imsize) + "_pairs_type:_"+str(args.pairs_type)
				 )

	if args.mini: checkpoint_path += "_mini_db"
	if args.pca: checkpoint_path += "_pca"
	if args.emb_proj: checkpoint_path += "_emb_proj"
	if args.pretrained: checkpoint_path += "_pretrained"
	if args.seed is not None: checkpoint_path += "_seed:_" + str(args.seed)

	checkpoint_path = os.path.join(args.directory,checkpoint_path)

	transform_train = augmentation("augment_train",args.imsize)
	transform_inference = augmentation("augment_inference")
	
	train_root = args.info_dir
	query_root = train_root

	train_infer_dataset = MET_database(root = train_root,mini = args.mini,
		transform = transform_inference,im_root = args.im_root)
	val_dataset = MET_queries(root = query_root,transform = transform_inference,
		im_root = args.im_root)

	#batch size on the inference loaders is 1 because we don't crop images at inference stage
	train_infer_loader = torch.utils.data.DataLoader(train_infer_dataset,
		batch_size=1,shuffle=False,num_workers=8,pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset,
		batch_size=1,shuffle=False,num_workers=8,pin_memory=True)

	if args.init_descr is not None:

		if os.path.isfile(args.init_descr+"/descriptors.pkl"):
			with open(args.init_descr+"/descriptors.pkl",'rb') as data:
				data_dict = pickle.load(data)
				train_descr = np.array(data_dict["train_descriptors"]).astype("float32")
				print("num of loaded train descr used for pca stats")
				print(np.shape(train_descr)[0])

		else:
			sys.exit("File {} does not exist".format(args.init_descr+"/descriptors.pkl"))

	else:
		print("extracting initial descriptors")
		train_descr = extract_embeddings(siamese_network(args.net,pooling = "gem",
			pretrained = args.pretrained).backbone.cuda(),train_infer_loader,ms = [1],msp = 1.0)


	if args.emb_proj: #in this case the backbone contains the optional FC layer

		if args.pca:
			PCA_stats = estimate_pca_whiten_with_shrinkage(train_descr,dimensions = np.shape(train_descr)[1])

	else:
		PCA_stats = None


	train_dataset = MET_pairs_dataset(root = train_root,mini = args.mini,
		transform = transform_train,pairs_type = args.pairs_type,
		train_descr = train_descr,im_root = args.im_root) #provide the Imagenet descriptors for the first epoch

	train_loader = torch.utils.data.DataLoader(train_dataset,
		batch_size=int(args.bsize),shuffle=True,num_workers=8,pin_memory=True)


	#print some statistics of the dataset
	print("Number of train pairs to be created: ")
	print(len(train_dataset))
	print("Number of train images in the inference train set: ")
	print(len(train_infer_dataset))
	print("Number of val images: ")
	print(len(val_dataset))
	print("Total number of classes in the database in the inference train set: ")
	print(len(np.unique(train_infer_dataset.targets)))


	#initialize the model and move it to the gpu
	model = siamese_network(args.net,pooling = "gem",pretrained = args.pretrained,
				emb_proj = args.emb_proj,init_emb_projector = PCA_stats).cuda()

	criterion = ContrastiveLoss(margin = args.margin).cuda()
	
	optimizer = optim.Adam(model.parameters(),lr = float(args.backbone_lr),
					weight_decay = args.wdecay)

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step, gamma=args.sched_gamma)

	start_epoch = 1


	if args.resume is not None:
				
		if os.path.isfile(args.resume):
			
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch'] + 1
			val_gap = checkpoint['val_gap']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])			
			scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step,
							 gamma=args.sched_gamma,last_epoch=checkpoint['epoch']+1)

		else:
			print("No checkpoint found at '{}'".format(args.resume))

	else:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step, gamma=args.sched_gamma)
		best_val_gap = 0.0
	
	

	if args.initeval:
		print("Backbone evaluation")
		print("Printing validation set knn evaluation metrics before train")
		
		val_gap,val_non_distr_gap,val_acc = validate(model.backbone,train_infer_loader,
			np.array(train_infer_dataset.targets),val_loader,np.array(val_dataset.targets))


	lr_backbone = optimizer.param_groups[0]['lr']
	print('Backbone lr: {:.2e}'.format(lr_backbone))

	#training loop	
	for epoch in range(start_epoch,int(args.epochs)+1):

		if epoch != start_epoch:
			print("creating pairs for epoch: "+str(epoch))
			train_dataset.create_epoch_pairs(train_descr)
	
		epoch_loss = train_contrastive_1epoch_virtual(model,criterion,optimizer,train_loader,epoch,args.vbsizemul)
		
		print("Epoch : " + str(epoch) + " , training phase finished, proceeding with validation set knn evaluation")


		#evaluate performance on validation set and simultaneous extraction of descriptors for the next epoch
		print("backbone evaluation")
		print("Printing validation set knn evaluation metrics for epoch: " + str(epoch))

		val_gap,val_non_distr_gap,val_acc,train_descr = validate(model.backbone,train_infer_loader,
								np.array(train_infer_dataset.targets),val_loader,
								np.array(val_dataset.targets),ret_train_descr = True)

		#Storing the descriptors for later use
		descriptors_dict = {}
		descriptors_dict["train_descriptors"] = train_descr

		if args.mini:
			filename = args.directory +"/train_mini_descriptors" + "_epoch:_{}.pkl".format(epoch)

		else:
			filename = args.directory +"/train_descriptors" + "_epoch:_{}.pkl".format(epoch)

		with open(filename, 'wb') as data:
			pickle.dump(descriptors_dict,data,protocol = pickle.HIGHEST_PROTOCOL)
			print("descriptors pickle file complete: {}".format(filename))


		if checkpoint_path is not None:

			save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),'val_gap': val_gap,
					'optimizer' : optimizer.state_dict()},checkpoint_path,epoch)
			
			print("Checkpoint saved at: " + checkpoint_path + "_epoch:_" + str(epoch))

		#adjust optimizer's learning rate
		scheduler.step()

		lr_backbone = optimizer.param_groups[0]['lr']
		print('Backbone lr: {:.2e}'.format(lr_backbone))




if __name__ == '__main__':
	main()