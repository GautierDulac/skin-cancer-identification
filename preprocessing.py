###Imports
import os
import torch
from torchvision import transforms,datasets


###Constants
data_dir = 'data'
batch_size_train=64
batch_size_val=5
shuffle_train=True
shuffle_val=False
num_workers=6


###Main function
def split_train_valid_sets(data_dir, batch_size_train, batch_size_val, shuffle_train, shuffle_val, num_workers):
    '''
    Provides an iterable over train set and validation set (by combining a dataset and a sampler)

    :param data_dir: directory containing the dataset
    :param batch_size_train: how many samples per batch to load from train set
    :param batch_size_val: how many samples per batch to load from validation set
    :param shuffle_train: set to True to have the train set data reshuffled at every epoch
    :param shuffle_val: set to True to have the validation set data reshuffled at every epoch
    :param num_workers: how many subprocesses to use for data loading
    :return:
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    vgg_format = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), vgg_format)
             for x in ['train', 'valid']}

    loader_train = torch.utils.data.DataLoader(dsets['train'], batch_size=batch_size_train, shuffle=shuffle_train, num_workers=num_workers)
    loader_valid = torch.utils.data.DataLoader(dsets['valid'], batch_size=batch_size_val, shuffle=shuffle_val, num_workers=num_workers)

    return loader_train, loader_valid





