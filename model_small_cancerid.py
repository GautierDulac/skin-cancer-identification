# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:54:15 2019

@author: Ben
"""
### Imports
import os
import torch
from torchvision import transforms, datasets, models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

###Constants
data_dir = 'data'
batch_size_train=64
batch_size_val=5
shuffle_train=True
shuffle_val=False
num_workers=6
loss_fn = nn.NLLLoss()
learning_rate = 1e-3
n_epochs = 50
epochs_jump = 2

###Load function
def split_train_valid_sets(batch_size_train, batch_size_val, shuffle_train, shuffle_val, num_workers=6):
    '''
    Provides an iterable over train set and validation set (by combining a dataset and a sampler) and the size of train and validation sets

    :param batch_size_train: how many samples per batch to load from train set
    :param batch_size_val: how many samples per batch to load from validation set
    :param shuffle_train: set to True to have the train set data reshuffled at every epoch
    :param shuffle_val: set to True to have the validation set data reshuffled at every epoch
    :param num_workers: how many subprocesses to use for data loading
    :return: loaders for train and validation sets and size of train and validation sets
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    vgg_format = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), vgg_format)
             for x in ['train', 'valid']}

    dset_sizes = {x: len(dsets[x]) for x in ['train', 'valid']}
    train_size = dset_sizes['train']
    valid_size = dset_sizes['valid']

    loader_train = torch.utils.data.DataLoader(dsets['train'], batch_size=batch_size_train, shuffle=shuffle_train, num_workers=num_workers)
    loader_valid = torch.utils.data.DataLoader(dsets['valid'], batch_size=batch_size_val, shuffle=shuffle_val, num_workers=num_workers)

    return train_size, valid_size, loader_train, loader_valid

### Model definition

class classifier(nn.Module):

    def __init__(self):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=18, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=18,out_channels=48, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features= 3528 , out_features=64)
        self.fc2 = nn.Linear(in_features= 64 , out_features=2)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 8)
        F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        self.fc2(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


### Training function

def train(model,data_loader,loss_fn,optimizer,n_epochs=1):
    model.train(True)
    loss_train = np.zeros(n_epochs)
    acc_train = np.zeros(n_epochs)
    r_train = np.zeros(n_epochs)
    for epoch_num in range(n_epochs):
        running_corrects = 0.0
        running_loss = 0.0
        running_positives = 0
        running_true_positives = 0
        size = 0
        for data in data_loader:
          
          inputs, classes = data
          bs = classes.size(0)
          inputs = inputs.to(device)
          classes = classes.to(device)
          outputs = model(inputs)
          loss = loss_fn(outputs,classes) 
          optimizer = optimizer_cl
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          _,preds = torch.max(outputs.data,1)
          running_loss+=loss
          running_corrects += torch.sum(preds==classes.data)

          size += bs
        epoch_loss = running_loss.item() / size
        epoch_acc = running_corrects.item() / size
        running_true_positives += torch.sum((classes.data == 1) & (preds == classes.data))
        running_positives += torch.sum(classes.data == 1)
        epoch_recall = running_true_positives.to('cpu').numpy() / running_positives.to('cpu').numpy()
        loss_train[epoch_num] = epoch_loss
        acc_train[epoch_num] = epoch_acc
        r_train[epoch_num] = epoch_recall
        print('Train - Loss: {:.4f} Acc: {:.4f} Rec: {:.4f}'.format(epoch_loss, epoch_acc, epoch_recall))
#    return loss_train, acc_train, r_train

### Test function

def test(model,data_loader):
    model.train(False)
    running_corrects = 0.0
    running_loss = 0.0
    running_positives = 0
    running_true_positives = 0
    size = 0
    i = 0
    for data in data_loader:
        inputs, classes = data
        inputs = inputs.to(device)
        classes = classes.to(device)
        bs = classes.size(0) 
        outputs = model(inputs).to(device)
        loss = loss_fn(outputs,classes.type(torch.cuda.LongTensor))
        _,preds = torch.max(outputs,1)
        running_corrects += torch.sum(preds == classes.data.type(torch.cuda.LongTensor))
        running_loss += loss.data
        running_true_positives += torch.sum((classes.data == 1) & (preds == classes.data))
        running_positives += torch.sum(classes.data == 1)
        predictions[i:i+len(classes)] = preds.to('cpu').numpy()
        all_classes[i:i+len(classes)] = classes.to('cpu').numpy()
        all_proba[i:i+len(classes), :] = outputs.data.to('cpu').numpy()
        i += len(classes)
        epoch_recall = running_true_positives.to('cpu').numpy() / running_positives.to('cpu').numpy()
        size += bs
        l_test, a_test = running_loss / size, running_corrects.item() / size
    print('Test - Loss: {:.4f} Acc: {:.4f} Rec: {:.4f}'.format(l_test, a_test, r_test))
    return predictions, all_proba, all_classes

### Training with early stoppping
    
def validation_model_preconvfeat(model , batch_size_train, batch_size_val, shuffle_train, shuffle_valid,
                                 batch_size_preconvfeat, num_workers, optim = torch.optim.Adam(model.parameters(), lr = learning_rate)):
    train_size, valid_size, loader_train, loader_valid = split_train_valid_sets(batch_size_train, batch_size_val, shuffle_train, shuffle_val, num_workers)
    train(model,loader_train,loss_fn,optimizer_cl,n_epochs)
    predictions, all_proba, all_classes = test(model,loader_valid)        
    return predictions, all_proba, all_classes