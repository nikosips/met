from torchvision import transforms



def augmentation(key,imsize = 500):
	'''Using ImageNet statistics for normalization.
	'''

	augment_dict = {

		"augment_train":
			transforms.Compose([
				transforms.RandomResizedCrop(imsize, scale=(0.7,1.0),ratio = (0.99,1/0.99)),
				transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
				transforms.RandomGrayscale(p=0.2),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				]),
		
		"augment_inference":
			transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				])

	}

	return augment_dict[key]

