###Imports
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import ImageFile

###Constants
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_vgg = models.vgg16(pretrained=True)
fpath = 'data/imagenet_class_index.json'

batch_size_preconvfeat = 128
num_epochs = 50
lr = 0.001



###Main function



#### from preprocessing part
dset_sizes, loader_train, loader_valid = split_train_valid_sets(data_dir, batch_size_train, batch_size_val, shuffle_train, shuffle_val, num_workers)
conv_feat_train, labels_train = preconvfeat(loader_train)
conv_feat_valid, labels_valid = preconvfeat(loader_valid)
dset_sizes_train = dset_sizes['train']
dset_sizes_valid = dset_sizes['valid']

loaderfeat_train = create_preconvfeat_loader(conv_feat_train, labels_train, batch_size_preconvfeat, shuffle_train)
loaderfeat_valid = create_preconvfeat_loader(conv_feat_valid, labels_valid, batch_size_preconvfeat, shuffle_valid)

#Define a neural network
model_vgg = models.vgg16(pretrained=True)
#Define a loss function
criterion = nn.NLLLoss()
#Define an otpimizer function
optimizer_vgg = torch.optim.SGD(model_vgg.classifier[6].parameters(), lr=lr)


train_model(model_vgg.classifier, dataloader=loaderfeat_train, size=dset_sizes_train, epochs=num_epochs,
            optimizer=optimizer_vgg)



predictions, all_proba, all_classes = validation_model(model_vgg.classifier, dataloader=valid_loader,
                                                 size=dset_sizes['valid'])


###Core functions
def preconvfeat(dataloader):
    '''
    Precomputes features extraction from VGG network.

    :param dataloader: dataloader
    :return: features extracted with VGG network and labels associated
    '''
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
    '''
    Creates a loader to extracted features from a neural network.

    :param conv_feat: features extracted with a network
    :param labels: labels associated to features extracted with a network
    :param batch_size_preconvfeat: how many samples per batch to load from the dataset containing extracted features
    :param shuffle: set to True to have the extracted features dataset reshuffled at every epoch
    :return: loaders for extracted features
    '''
    dtype = torch.float
    datasetfeat = [[torch.from_numpy(f).type(dtype), torch.tensor(l).type(torch.long)] for (f, l) in
                   zip(conv_feat, labels)]
    datasetfeat = [(inputs.reshape(-1), classes) for [inputs, classes] in datasetfeat]
    loaderfeat = torch.utils.data.DataLoader(datasetfeat, batch_size=batch_size_preconvfeat, shuffle=shuffle)
    return loaderfeat


def train_model(model, dataloader, size, epochs=1, optimizer=None):
    '''
    Computes loss and accuracy on validation test

    :param model: neural network model
    :param dataloader: train loader
    :param size: number of images in the dataset
    :param epochs: number of epochs
    :param optimizer: optimization algorithm
    :return:
    '''
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        for inputs, classes in dataloader:
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs.data, 1)
            # statistics
            running_loss += loss.data.item()
            running_corrects += torch.sum(preds == classes.data)
        epoch_loss = running_loss / size
        epoch_acc = running_corrects.data.item() / size
        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))



def validation_model(model, dataloader, size):
    '''
    Computes loss and accuracy on validation test

    :param model: neural network model
    :param dataloader: test loader
    :param size: number of images in the dataset
    :return: predictions, probabilities and classes for the validation set
    '''
    model.eval()
    predictions = np.zeros(size)
    all_classes = np.zeros(size)
    all_proba = np.zeros((size,2))
    i = 0
    running_loss = 0.0
    running_corrects = 0
    for inputs, classes in dataloader:
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        loss = criterion(outputs,classes)
        _, preds = torch.max(outputs.data,1)
            # statistics
        running_loss += loss.data.item()
        running_corrects += torch.sum(preds == classes.data)
        predictions[i:i+len(classes)] = preds.to('cpu').numpy()
        all_classes[i:i+len(classes)] = classes.to('cpu').numpy()
        all_proba[i:i+len(classes), :] = outputs.data.to('cpu').numpy()
        i += len(classes)
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.data.item() / size
    print('Loss: {:.4f} Acc: {:.4f}'.format(
                     epoch_loss, epoch_acc))
    return predictions, all_proba, all_classes