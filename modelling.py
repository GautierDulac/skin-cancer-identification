###Imports
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from PIL import ImageFile
import json


###Constants
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_vgg = models.vgg16(pretrained=True)
fpath = 'data/imagenet_class_index.json'

batch_size_preconvfeat = 128

###Main function

model_vgg = models.vgg16(pretrained=True)

conv_feat_train,labels_train = preconvfeat(loader_train)
conv_feat_valid,labels_valid = preconvfeat(loader_valid)

loaderfeat_train = create_preconvfeat_loader(conv_feat_train, labels_train, batch_size_preconvfeat, shuffle_train)
loaderfeat_valid = create_preconvfeat_loader(conv_feat_valid, labels_valid, batch_size_preconvfeat, shuffle_valid)

train_model(model_vgg.classifier,dataloader=loaderfeat_train,size=dset_sizes['train'],epochs=50,optimizer=optimizer_vgg)
predictions, all_proba, all_classes = test_model(model_vgg.classifier,dataloader=valid_loader,size=dset_sizes['valid'])

###Core functions
def preconvfeat(dataloader):
    conv_features = []
    labels_list = []
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        x = model_vgg.features(inputs)
        conv_features.extend(x.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
    conv_features = np.concatenate([[feat] for feat in conv_features])
    return conv_features, labels_list


def create_preconvfeat_loader(conv_feat, labels, batch_size_preconvfeat, shuffle):
    dtype=torch.float
    datasetfeat_train = [[torch.from_numpy(f).type(dtype),torch.tensor(l).type(torch.long)] for (f,l) in zip(conv_feat,labels)]
    datasetfeat_train = [(inputs.reshape(-1), classes) for [inputs,classes] in datasetfeat_train]
    loaderfeat_train = torch.utils.data.DataLoader(datasetfeat_train, batch_size=batch_size_preconvfeat, shuffle=shuffle)
    return loaderfeat_train

