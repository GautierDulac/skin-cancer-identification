###Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import ImageFile
from preprocessing import split_train_valid_sets
from visualizing import training_visualisation

###Constants
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fpath = 'data/imagenet_class_index.json'

batch_size_preconvfeat = 128


###Main function
def main_model():
    return


###Core functions
def create_preconvfeat_loader(dataloader, model, batch_size_preconvfeat, shuffle):
    '''
    Precomputes features extraction and creates a loader to extract features.
    
    :param dataloader: dataloader
    :param model: model from which we extract feature
    :param batch_size_preconvfeat: how many samples per batch to load from the dataset containing extracted features
    :param shuffle: set to True to have the extracted features dataset reshuffled at every epoch
    :return: loaders for extracted features
    '''
    dtype = torch.float
    conv_features = []
    labels_list = []
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        x = model.features(inputs)
        conv_features.extend(x.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())

    print(x.size())

    conv_features = np.concatenate([[feat] for feat in conv_features])
    datasetfeat = [[torch.from_numpy(f).type(dtype), torch.tensor(l).type(torch.long)] for (f, l) in
                   zip(conv_features, labels_list)]
    datasetfeat = [(inputs.reshape(-1), classes) for [inputs, classes] in datasetfeat]
    print(len(datasetfeat))
    print(len(datasetfeat[0]))
    print(datasetfeat[0][0].size())
    loaderfeat = torch.utils.data.DataLoader(datasetfeat, batch_size=batch_size_preconvfeat, shuffle=shuffle)

    return loaderfeat


def train_model(model, dataloader, size, epochs=1, optimizer=None, criterion=None, model_select=None, lr=None):
    '''
    Computes loss and accuracy on validation test

    :param num_epochs:
    :param lr:
    :param model_select:
    :param criterion:
    :param model: neural network model
    :param dataloader: train loader
    :param size: number of images in the dataset
    :param epochs: number of epochs
    :param optimizer: optimization algorithm
    :return:
    '''
    model.train()
    loss_list = []
    acc_list = []
    recall_list = []
    precision_list = []

    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        running_true_positives = 0
        running_positives = 0
        running_pred_positives = 0
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
            running_true_positives += torch.sum((classes.data == 1) & (preds == classes.data))
            running_pred_positives += torch.sum(preds == 1)
            running_positives += torch.sum(classes.data == 1)
        epoch_loss = running_loss / size
        epoch_acc = running_corrects.to('cpu').numpy() / size
        epoch_recall = running_true_positives.to('cpu').numpy() / running_positives.to('cpu').numpy()
        epoch_precision = running_true_positives.to('cpu').numpy() / running_pred_positives.to('cpu').numpy()
        print('Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f}'.format(
            epoch_loss, epoch_acc, epoch_precision, epoch_recall))
        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)
        precision_list.append(epoch_precision)
        recall_list.append(epoch_recall)
    training_visualisation(loss_list, acc_list, precision_list, recall_list, model_select, lr, epochs)


def multi_plots(loss_list, recall_list):
    plt.plot(loss_list)
    plt.plot(recall_list)


def validation_model(model, dataloader, size, criterion):
    '''
    Computes loss and accuracy on validation test

    :param criterion:
    :param model: neural network model
    :param dataloader: test loader
    :param size: number of images in the dataset
    :return: predictions, probabilities and classes for the validation set
    '''
    model.eval()

    predictions = np.zeros(size)
    all_classes = np.zeros(size)
    all_proba = np.zeros((size, 2))
    i = 0
    running_loss = 0.0
    running_corrects = 0
    running_true_positives = 0
    running_positives = 0
    running_pred_positives = 0
    for inputs, classes in dataloader:
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, classes)
        _, preds = torch.max(outputs.data, 1)
        # statistics
        running_loss += loss.data.item()
        running_corrects += torch.sum(preds == classes.data)
        running_true_positives += torch.sum((classes.data == 1) & (preds == classes.data))
        running_pred_positives += torch.sum(preds == 1)
        running_positives += torch.sum(classes.data == 1)
        predictions[i:(i + len(classes))] = preds.to('cpu').numpy()
        all_classes[i:i + len(classes)] = classes.to('cpu').numpy()
        all_proba[i:i + len(classes), :] = outputs.data.to('cpu').numpy()
        i += len(classes)
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.to('cpu').numpy() / size
    epoch_recall = running_true_positives.to('cpu').numpy() / running_positives.to('cpu').numpy()
    epoch_precision = running_true_positives.to('cpu').numpy() / running_pred_positives.to('cpu').numpy()
    print('Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f}'.format(
        epoch_loss, epoch_acc, epoch_precision, epoch_recall))
    return predictions, all_proba, all_classes


def validation_model_preconvfeat(model, batch_size_train, batch_size_val, shuffle_train, shuffle_valid, num_workers,
                                 optim=None, criterion=None, model_select=None, num_epochs=None, lr=None):
    '''
    Computes predictions, probabilities and classes for validation set with precomputed extracted features

    :param model: neural network model
    :param batch_size_train: how many samples per batch to load from train set
    :param batch_size_val: how many samples per batch to load from validation set
    :param shuffle_train: set to True to have the train set data reshuffled at every epoch
    :param shuffle_val: set to True to have the validation set data reshuffled at every epoch
    :param batch_size_preconvfeat: how many samples per batch to load from the dataset containing extracted features
    :param num_workers: how many subprocesses to use for data loading
    :return: predictions, probabilities and classes for the validation set
    '''

    train_size, valid_size, loader_train, loader_valid = split_train_valid_sets(batch_size_train,
                                                                                batch_size_val, shuffle_train,
                                                                                shuffle_valid, num_workers)

    if model_select in [1, 3]:
        loader_train = create_preconvfeat_loader(loader_train, model, batch_size_preconvfeat, shuffle_train)
        loader_valid = create_preconvfeat_loader(loader_valid, model, batch_size_preconvfeat, shuffle_valid)
        print(model)
        model = model.classifier
        print(model)
        for data in loader_train:
            inputs, = data
            print(inputs.size())
            break
        print("Preconvfeat done")

    train_model(model, dataloader=loader_train, size=train_size, epochs=num_epochs, optimizer=optim,
                criterion=criterion, model_select=model_select, lr=lr)

    predictions, all_proba, all_classes = validation_model(model, dataloader=loader_valid, size=valid_size,
                                                           criterion=criterion)

    return predictions, all_proba, all_classes
