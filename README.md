## The Met dataset

This is the repository of the Met dataset. The official webpage of the dataset can be found [here](http://cmp.felk.cvut.cz/met/).

<img src= "http://cmp.felk.cvut.cz/met/img/1.jpg" width=\textwidth/>

---

### What is it?

The code provides examples for the following:

1. How to use the dataset.
1. How to evaluate your own method.
1. How to reproduce some of the baselines presented in the NeurIPS paper.

---

### Prerequisites

In order to run the code you will need:

1. Python3
1. NumPy
1. Faiss library for efficient similarity search (faiss-gpu)
1. PyTorch deep learning framework
1. The Met dataset (images + ground truth) from the official website

---

### Embedding models

We provide models for descriptor extraction. You can download them [here](http://cmp.felk.cvut.cz/met/#models).

---

### Pre-extracted descriptors

We provide pre-extracted descriptors. You can download them [here](http://cmp.felk.cvut.cz/met/#descriptors).

---

### Usage

Navigate (```cd```) to ```[YOUR_MET_ROOT]/met```. ```[YOUR_MET_ROOT]``` is where you have cloned the github repository. 

<details>

  <summary><b>Descriptor extraction</b></summary><br/>
  
  Example script for extracting descriptors for the images of the Met dataset is located in ```code/examples/extract_descriptors.py```

  For detailed explanation of the options run:  
  ```
  python3 -m code.examples.extract_descriptors -h
  ```

</details>

<details>

  <summary><b>kNN classifier & evaluation</b></summary><br/>
  
  Example evaluation script of pre-extracted descriptors with the non-parametric classifier is located in ```code/examples/knn_eval.py```

  For detailed explanation of the options run:  
  ```
  python3 -m code.examples.knn_eval -h
  ```

  Example (using ground truth and descriptors downloaded from our website, after unzipping both):  
  ```
  python -m code.examples.knn_eval [YOUR_DESCRIPTOR_DIR] --autotune --info_dir [YOUR_GROUND_TRUTH_DIR]
  ```

</details>

<details>
  
  <summary><b>Training with contrastive loss</b></summary><br/>

  Example training script for training the embedding model with contrastive loss on the Met training set is located in ```code/examples/train_contrastive.py```. The trained network can be used for descriptor extraction and kNN classification.

  For detailed explanation of the options run:  
  ```
  python3 -m code.examples.train_contrastive -h
  ```

</details>


---


### Reproducing results from the paper

We provide an example for each one of the uses of the provided code. Different combinations should work in the same manner.

Navigate (```cd```) to ```[YOUR_MET_ROOT]/met```. ```[YOUR_MET_ROOT]``` is where you have cloned the github repository. 

<details>

  <summary><b>Descriptor Extraction</b></summary><br/>
  
  Extracting r18SWSL_con-syn+real-closest descriptors using the trained model provided in our website. Ground truth is stored in ```./data/ground_truth/``` and images are stored in ```./data/images```, after both have been extracted from the .zip files. The checkpoint of the model to be loaded is stored in ```./data/models/r18SWSL_con-syn+real-closest```. The descriptors will be stored in ```./data/descriptors``` after the extraction:
  ```
  python -m code.examples.extract_descriptors ./data/descriptors --info_dir ./data/ground_truth --im_root ./data/ --net r18_contr_loss_gem_fc_swsl --gpuid 0 --netpath ./data/models/r18SWSL_con-syn+real-closest --ms
  ```

</details>

<details>

  <summary><b>kNN classifier & evaluation</b></summary><br/>

  Evaluating r18SWSL_con-syn+real-closest descriptors. Ground truth is stored in ```./data/ground_truth/``` and descriptors are stored in ```./data/descriptors/r18SWSL_con-syn+real-closest```, after both have been extracted from the .zip files:  
  ```
  python -m code.examples.knn_eval ./data/descriptors/r18SWSL_con-syn+real-closest/ --autotune --info_dir ./data/ground_truth/
  ```

</details>

<details>
  
  <summary><b>Training with contrastive loss</b></summary><br/>

  Training a pretrained on ImageNet ResNet18 backbone (FC layer included, initialized with PCAw) using the contrastive loss and the con-syn+real-closest pairs (described in the paper). Ground truth is stored in ```./data/ground_truth/``` and images are stored in ```./data/images```, after both have been extracted from the .zip files. The checkpoints of the model will be stored in ```./data/models/r18SWSL_con-syn+real-closest```:
  ```
  python -m code.examples.train_contrastive --seed 0 --pretrained --pairs_type new_pos+new_neg ./data/models/r18SWSL_con-syn+real-closest --emb_proj --pca --info_dir ./data/ground_truth --im_root ./data/ --gpuid 0
  ```

</details>


---

### State

Repository is under update. Feel free to give feedback, by sending an email to: ypsilnik@fel.cvut.cz

---
