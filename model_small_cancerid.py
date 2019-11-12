### Imports
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import split_train_valid_sets
from modelling import train_model
from modelling import validation_model

###Constants


###Load function


### Model definition

class classifier(nn.Module):

    def __init__(self):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=18, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=48, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=3528, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
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
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


### main model function

def validation_model_preconvfeat_2(model, batch_size_train, batch_size_val, shuffle_train, shuffle_val, num_workers,
                                   optim=None, criterion=None, model_select=None, num_epochs=None, lr=None):
    """

    :param model:
    :param batch_size_train:
    :param batch_size_val:
    :param shuffle_train:
    :param shuffle_val:
    :param num_workers:
    :param optim:
    :param criterion:
    :param model_select:
    :param num_epochs:
    :param lr:
    :return:
    """

    for param in model.parameters():
        param.requires_grad = True

    train_size, valid_size, loader_train, loader_valid = split_train_valid_sets(batch_size_train, batch_size_val,
                                                                                shuffle_train, shuffle_val, num_workers)
    train_model(model, loader_train, batch_size_train, epochs=num_epochs, optimizer=optim, criterion=criterion,
                model_select=model_select, lr=lr)
    predictions, all_proba, all_classes = validation_model(model, loader_valid, batch_size_val, criterion)
    return predictions, all_proba, all_classes
