import os
import os.path
import json
import pickle
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader

from code.utils.train_utils import *




class MET_queries(VisionDataset):

    def __init__(
            self,
            root: str = ".",
            test: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            im_root = None
    ) -> None:
        super().__init__(root, transform=transform,
                            target_transform=target_transform)

        if test:
            fn = "testset.json"
        else:
            fn = "valset.json"

        with open(os.path.join(self.root, fn)) as f:
            data = json.load(f)

        samples = []
        targets = []

        for e in data:
        
            samples.append(e['path'])
            if "MET_id" in e:
                targets.append(int(e['MET_id']))
            else:
                targets.append(-1)

        self.loader = loader
        self.samples = samples
        self.targets = targets

        assert len(self.samples) == len(self.targets)

        self.im_root = im_root

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if self.im_root is not None:
            path = os.path.join(self.im_root, "images/" + self.samples[index])            

        else:
            path = os.path.join(os.path.dirname(self.root), "images/" + self.samples[index])
        
        target = self.targets[index]

        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self) -> int:
        
        return len(self.samples)



class MET_database(VisionDataset):

    def __init__(
            self,
            root: str = ".",
            mini: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            im_root = None
    ) -> None:
        super().__init__(root, transform=transform,
                            target_transform=target_transform)

        fn = "MET_database.json"

        if mini:
            fn = "mini_"+fn

        with open(os.path.join(self.root, fn)) as f:
            data = json.load(f)

        samples = []
        targets = []

        for e in data:
            samples.append(e['path'])
            targets.append(int(e['id']))

        self.loader = loader
        self.samples = samples
        self.targets = targets

        assert len(self.samples) == len(self.targets)

        self.im_root = im_root


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        if self.im_root is not None:
            path = os.path.join(self.im_root, "images/" + self.samples[index])            

        else:
            path = os.path.join(os.path.dirname(self.root), "images/" + self.samples[index])

        target = self.targets[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self) -> int:
        
        return len(self.samples)



class MET_pairs_dataset(VisionDataset):
    '''
    Dataset class that provides pairs of images from the Met training set, along with a label.
    The label is 1 for positive (same class) pair, 0 for negative (different class).
    To be used with Contrastive learning on the Met dataset.
    Can also be used with SimSiam method of training that only requires positive pairs.
    '''

    def __init__(
            self,
            root: str = ".",
            mini: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            pairs_type = None,
            train_descr = None,
            im_root = None
    ) -> None:

        super().__init__(root, transform=transform,
                            target_transform=target_transform)

        fn = "MET_database.json"
        if mini:
            fn = "mini_"+fn

        with open(os.path.join(self.root, fn)) as f:
            data = json.load(f)

        samples = []
        targets = []

        for e in data:
            samples.append(e['path'])
            targets.append(int(e['id']))

        self.loader = loader
        self.samples = samples
        self.targets = targets

        assert len(self.samples) == len(self.targets)

        self.pairs = []
        self.pair_targets = []

        self.root = root
        self.im_root = im_root

        self.num_classes = len(np.unique(np.array(self.targets)))
        print("num classes in the database before turning into pairs: " + str(self.num_classes))

        self.pairs_type = pairs_type

        print("creating pairs for the first epoch by the pretrained descriptors")
        self.create_epoch_pairs(train_descr)



    def __getitem__(self,index):

        if self.im_root is not None:
            path1 = os.path.join(self.im_root,"images/"+self.pairs[index][0])
            path2 = os.path.join(self.im_root,"images/"+self.pairs[index][1])

        else:
            path1 = os.path.join(os.path.dirname(self.root),"images/"+self.pairs[index][0])
            path2 = os.path.join(os.path.dirname(self.root),"images/"+self.pairs[index][1])

        pair_target = self.pair_targets[index]
        
        sample1 = self.loader(path1)
        sample2 = self.loader(path2)
        
        if self.transform is not None:
            sample1 = self.transform(sample1) #transformation is random, so it is different for each of the two samples
            sample2 = self.transform(sample2)
        
        if self.target_transform is not None:
            pair_target = self.target_transform(pair_target)

        return (sample1,sample2), pair_target


    def __len__(self) -> int:

        if self.pairs_type == "sim_siam_pos": #for sim siam method
            return len(self.samples)

        elif self.pairs_type == "pos+new_neg":
            return len(self.samples) + len(self.samples)

        elif self.pairs_type == "new_pos+new_neg":
            return len(self.samples) + len(self.samples)

        elif self.pairs_type == "sim_siam_pos+new_neg":
            return len(self.samples) + len(self.samples)


    def create_epoch_pairs(self,train_descr = None):

        print("creating pairs")

        self.samples2 = np.array(self.samples) #copy them in order not to change them
        self.targets2 = np.array(self.targets)
        
        self.pairs = [] #create the list from scratch every time create epoch pairs is called
        self.pair_targets = []
    
        #positive pairs
        print("creating positive pairs")

        if self.pairs_type == "sim_siam_pos" or self.pairs_type == "sim_siam_pos+new_neg":

            for i,sample in enumerate(self.samples2):
                    self.pairs.append((sample,sample))
                    self.pair_targets.append(1) #1 to indicate positive pair

        #pos without mining
        if self.pairs_type == "pos+new_neg":

            class_idx_dict = create_class_idx_dict(self.targets2)

            for i,sample in enumerate(self.samples2):
                same_class_sample_idxs = list(class_idx_dict[self.targets2[i]])

                if len(same_class_sample_idxs) == 1:
                    self.pairs.append((sample,sample))
                    self.pair_targets.append(1) #1 to indicate positive pair
                
                else:
                    index2 = np.random.choice(same_class_sample_idxs,1)[0]
                    self.pairs.append((sample,self.samples2[index2]))
                    self.pair_targets.append(1)


        #pos with mining
        if self.pairs_type == "new_pos+new_neg":

            class_idx_dict = create_class_idx_dict(self.targets2)
            
            for i,sample in enumerate(self.samples2):

                same_class_sample_idxs = list(class_idx_dict[self.targets2[i]])

                if len(same_class_sample_idxs) == 1:
                    self.pairs.append((sample,sample))
                    self.pair_targets.append(1) #1 to indicate positive pair
                
                else:
                    index2 = mine_positive(i,same_class_sample_idxs,train_descr)
                    self.pairs.append((sample,self.samples2[index2]))
                    self.pair_targets.append(1)


        print("number of positive pairs created: " + str(len(self.pairs)))


        #negative pairs
        if self.pairs_type == "pos+new_neg" or self.pairs_type == "new_pos+new_neg" or self.pairs_type == "sim_siam_pos+new_neg":

            print("creating negative pairs")

            negatives = mine_negatives(self.samples2,self.root,train_descr,self.targets2)
            
            #negatives is a list with the negative corresponding to each sample
            for i,image in enumerate(self.samples2):
                self.pairs.append((image,negatives[i]))
                self.pair_targets.append(0)


        print("total number of pairs created: " + str(len(self.pairs)))
