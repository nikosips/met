import os
import sys
import pickle
import json
import numpy as np
import argparse
from collections import OrderedDict

import torch
from torchvision import transforms
from torch.utils.model_zoo import load_url

from code.utils.datasets import *
from code.utils.utils import *
from code.networks.backbone import *
from code.networks.SiameseNet import *
from code.utils.augmentations import augmentation


'''Script for the extraction of descriptors for the Met dataset given a (pretrained) backbone.
'''


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('directory', metavar='EXPORT_DIR',help='destination where descriptors will be saved')
	parser.add_argument('--gpuid', default=0, type=int) #id of the gpu in your machine
	parser.add_argument('--net', default='r18INgem')
	parser.add_argument('--netpath', default=None) #optional
	parser.add_argument('--ms', action='store_true') #multiscale descriptors
	parser.add_argument('--mini', action='store_true') #use the mini database
	parser.add_argument('--queries_only', action='store_true')
	parser.add_argument('--trained_on_mini', action='store_true') #if your model has a classification head for the mini dataset
	parser.add_argument('--info_dir',default=None, type=str, help = 'directory where ground truth is stored')
	parser.add_argument('--im_root',default=None, type=str, help = 'directory where images are stored')
	
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)

	# folder name
	network_variant = args.net
	exp_name = network_variant
	if args.ms:
		exp_name+=("_ms")
	else:
		exp_name+=("_ss")
	if args.mini:
		exp_name+=("_mini")
	if args.queries_only:
		exp_name+=("_queries_only")
	
	exp_dir = args.directory+"/"+exp_name+"/"
	
	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir, exist_ok=True)
	
	print("Will save descriptors in {}".format(exp_dir))
	
	extraction_transform = augmentation("augment_inference")

	train_root = args.info_dir

	if not args.queries_only:
		train_dataset = MET_database(root = train_root,mini = args.mini,transform = extraction_transform,im_root = args.im_root)
	

	if args.trained_on_mini:
		num_classes = 33501

	else:
		num_classes = 224408

	query_root = train_root

	test_dataset = MET_queries(root = query_root,test = True,transform = extraction_transform,im_root = args.im_root)
	val_dataset = MET_queries(root = query_root,transform = extraction_transform,im_root = args.im_root)

	if not args.queries_only:
		train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
		print("Number of train images: {}".format(len(train_dataset)))

	test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
	print("Number of test images: {}".format(len(test_dataset)))
	print("Number of val images: {}".format(len(val_dataset)))


	#initialization of the global descriptor extractor model
	if args.netpath is not None:
		
		if network_variant == 'r18_contr_loss_gem':
			model = siamese_network("resnet18",pooling = "gem",pretrained = False)
			print("loading weights from checkpoint")
			model.load_state_dict(torch.load(args.netpath)['state_dict'])
			net = model.backbone

		elif network_variant == 'r18_contr_loss_gem_fc':
			model = siamese_network("resnet18",pooling = "gem",pretrained = False,
				emb_proj = True)
			model.backbone.projector.bias.data = model.backbone.projector.bias.data.unsqueeze(0)
			print("loading weights from checkpoint")
			model.load_state_dict(torch.load(args.netpath)['state_dict'])
			net = model.backbone

		elif network_variant == 'r18_contr_loss_gem_fc_swsl':
			model = siamese_network("r18_sw-sup",pooling = "gem",pretrained = False,
				emb_proj = True)
			model.backbone.projector.bias.data = model.backbone.projector.bias.data.unsqueeze(0)
			print("loading weights from checkpoint")
			model.load_state_dict(torch.load(args.netpath)['state_dict'])
			net = model.backbone
			
		else:
			raise ValueError('Unsupported  architecture: {}!'.format(network_variant))

	else:
		
		if network_variant == 'r18INgem':
			net = Embedder("resnet18",gem_p = 3.0,pretrained_flag = True)
		
		elif network_variant == 'r50INgem_caffe': #we use this version because it has the weights from caffe library which perform better
			net_params = {'architecture':"resnet50",'pooling':"gem",'pretrained':True,'whitening':False}	
			net = init_network(net_params)

		elif network_variant == 'r50INgem': #pytorch weights
			net = Embedder('resnet50',gem_p = 3.0,pretrained_flag = True,projector = False)

		elif network_variant == 'r50_swav_gem':
			model = torch.hub.load('facebookresearch/swav','resnet50')
			net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
			net.meta = {
				'architecture' : "resnet50", 
				'pooling' : "gem",
				'mean' : [0.485, 0.456, 0.406], #imagenet statistics
				'std' : [0.229, 0.224, 0.225],
				'outputdim' : 2048,
			}

		elif network_variant == 'r50_SIN_gem':

			model = torchvision.models.resnet50(pretrained=False)
			checkpoint = model_zoo.load_url('https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar')
			model = torch.nn.DataParallel(model)
			model.load_state_dict(checkpoint['state_dict'])
			model = model.module
			net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
			net.meta = {
				'architecture' : "resnet50", 
				'pooling' : "gem",
				'mean' : [0.485, 0.456, 0.406], #imagenet statistics
				'std' : [0.229, 0.224, 0.225],
				'outputdim' : 2048,
			}
		
		elif network_variant == 'r18_sw-sup_gem':
			model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')
			net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
			net.meta = {
				'architecture' : "resnet18", 
				'pooling' : "gem",
				'mean' : [0.485, 0.456, 0.406], #imagenet statistics
				'std' : [0.229, 0.224, 0.225],
				'outputdim' : 512,
			}

		elif network_variant == 'r50_sw-sup_gem':
			model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
			net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
			net.meta = {
				'architecture' : "resnet50", 
				'pooling' : "gem",
				'mean' : [0.485, 0.456, 0.406], #imagenet statistics
				'std' : [0.229, 0.224, 0.225],
				'outputdim' : 2048,
			}

		else:
			raise ValueError('Unsupported  architecture: {}!'.format(network_variant))


	net.cuda()

	if args.ms:
		#multi-scale case
		scales = [1, 1/np.sqrt(2), 1/2]

	else:
		#single-scale case
		scales = [1]

	print("Starting the extraction of the descriptors")
	
	if not args.queries_only:
		train_descr = extract_embeddings(net,train_loader,ms = scales,msp = 1.0,print_freq=20000)
		print("Train descriptors finished...")
	
	test_descr = extract_embeddings(net,test_loader,ms = scales,msp = 1.0,print_freq=5000)
	print("Test descriptors finished...")
	val_descr = extract_embeddings(net,val_loader,ms = scales,msp = 1.0,print_freq=1000)
	print("Val descriptors finished...")

	descriptors_dict = {}

	if not args.queries_only:
		descriptors_dict["train_descriptors"] = np.array(train_descr).astype("float32")

	descriptors_dict["test_descriptors"] = np.array(test_descr).astype("float32")
	descriptors_dict["val_descriptors"] = np.array(val_descr).astype("float32")

	#save descriptors
	with open(exp_dir+"descriptors.pkl", 'wb') as data:
		pickle.dump(descriptors_dict,data,protocol = pickle.HIGHEST_PROTOCOL)
		print("descriptors pickle file complete: {}".format(exp_dir+"descriptors.pkl"))



if __name__ == '__main__':
	main()