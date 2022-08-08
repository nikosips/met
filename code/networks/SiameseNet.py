import torch.nn as nn

from code.networks.backbone import Embedder




class siamese_network(nn.Module):
	'''Network architecture for contrastive learning.
	'''

	def __init__(self,backbone,pooling = "gem",pretrained = True,
					emb_proj = False,init_emb_projector= None):

		super(siamese_network,self).__init__()

		net = Embedder(backbone,gem_p = 3.0,pretrained_flag = pretrained,
						projector = emb_proj,init_projector = init_emb_projector) 

		self.backbone = net	#the backbone produces l2 normalized descriptors


	def forward(self,augs1,augs2):

		descriptors_left = self.backbone(augs1)
		descriptors_right = self.backbone(augs2)

		return descriptors_left,descriptors_right