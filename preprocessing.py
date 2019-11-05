###Imports
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torchvision import transforms,datasets


###Constants
data_dir = 'data'

###Main function


###Core functions
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

vgg_format = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), vgg_format)
         for x in ['train', 'valid']}


dset_sizes = {x: len(dsets[x]) for x in ['train', 'valid']}

dset_classes = dsets['train'].classes

loader_train = torch.utils.data.DataLoader(dsets['train'], batch_size=64, shuffle=True, num_workers=6)

loader_valid = torch.utils.data.DataLoader(dsets['valid'], batch_size=5, shuffle=False, num_workers=6)

count = 1
for data in loader_valid:
    print(count, end=',')
    if count == 1:
        inputs_try,labels_try = data
    count +=1


