import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models




OUTPUT_DIM = {
	'resnet18'              :  512,
	'resnet50'              : 2048,
	'r18_sw-sup'             : 512,
}



class GeM(nn.Module):
	'''Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
	'''
	
	def __init__(self, p=3, eps=1e-6):
		super(GeM,self).__init__()
		self.p = Parameter(torch.ones(1)*p)
		self.eps = eps

	def forward(self, x):
		return gem(x, p=self.p, eps=self.eps)
		
	def __repr__(self):
		return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'



def gem(x, p=3, eps=1e-6):
	'''Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
	'''

	return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)



class Embedder(nn.Module):
	'''Class that implements a descriptor extractor as a (fully convolutional backbone -> pooling -> l2 normalization).
	Optionally followed by a FC layer (fully convolutional backbone -> pooling -> l2 normalization -> FC -> l2 normalization)
	that can be initialized with the result of PCAw.
	'''

	def __init__(self,architecture,gem_p = 3,pretrained_flag = True,projector = False,init_projector = None):
		'''The FC layer is called projector.
		'''

		super(Embedder, self).__init__()

		if architecture == "r18_sw-sup": 
			#r18 facebook pretrained model (https://github.com/facebookresearch/semi-supervised-ImageNet1K-models)

			network = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models','resnet18_swsl')
			self.backbone = nn.Sequential(*list(network.children())[:-2])

		else:

			#load the base model from PyTorch's pretrained models (imagenet pretrained)
			network = getattr(models,architecture)(pretrained=pretrained_flag)

			#keep only the convolutional layers, ends with relu to get non-negative descriptors
			if architecture.startswith('resnet'):
				self.backbone = nn.Sequential(*list(network.children())[:-2])

			elif architecture.startswith('alexnet'):
				self.backbone = nn.Sequential(*list(network.features.children())[:-1])

		#spatial pooling layer
		self.pool = GeM(p = gem_p)

		#normalize on the unit-hypershpere
		#self.norm = L2N()
		self.norm = F.normalize

		#information about the network
		self.meta = {
			'architecture' : architecture, 
			'pooling' : "gem",
			'mean' : [0.485, 0.456, 0.406], #imagenet statistics for imagenet pretrained models
			'std' : [0.229, 0.224, 0.225],
			'outputdim' : OUTPUT_DIM[architecture],
		}


		if projector:
			print("using FC layer in the backbone")
			self.projector = nn.Linear(self.meta['outputdim'],self.meta['outputdim'],bias = True)

			if init_projector is not None:

				print("initialising the backbone's project layer")

				self.projector.weight.data = torch.transpose(torch.Tensor(init_projector[1]),0,1)
				self.projector.bias.data = -torch.matmul(torch.Tensor(init_projector[0]),torch.Tensor(init_projector[1]))

		else:
			self.projector = None


	def forward(self, img):
		'''
		Output has shape: batch size x descriptor dimension
		'''

		x = self.norm(self.pool(self.backbone(img))).squeeze(-1).squeeze(-1)

		if self.projector is None:
			return x

		else:
			return self.norm(self.projector(x))



def extract_ss(net, input):
	'''Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
	'''

	return net(input).cpu().data.squeeze()



def extract_ms(net, input, ms, msp):
	'''Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
	'''
	
	v = torch.zeros(net.meta['outputdim'])
	
	for s in ms:

		if s == 1:
			input_t = input.clone()
		
		else:    
			input_t = nn.functional.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
		
		v += net(input_t).pow(msp).cpu().data.squeeze()
		
	v /= len(ms)
	v = v.pow(1./msp)
	v /= v.norm()

	return v