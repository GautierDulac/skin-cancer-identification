###Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


###Constants


###Main function


###Core functions
def imshow(inp, title=None):
    """

    :param inp:
    :param title:
    :return:
    """
    #   Imshow for Tensor.
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = np.clip(std * inp + mean, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def final_visualisation(predictions, all_classes, dsets):
    """

    :param predictions:
    :param all_classes:
    :param dsets:
    :return:
    """
    # Number of images to view for each visualization task
    n_view = 8
    correct = np.where(predictions == all_classes)[0]
    from numpy.random import random, permutation
    idx = permutation(correct)[:n_view]
    loader_correct = torch.utils.data.DataLoader([dsets['valid'][x] for x in idx], batch_size=n_view, shuffle=True)
    for data in loader_correct:
        inputs_cor, labels_cor = data
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs_cor)
        imshow(out, title=[l.item() for l in labels_cor])
    return ()

