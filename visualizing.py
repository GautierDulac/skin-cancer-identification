###Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


###Constants


###Main function


###Core functions
def imshow(inp, title=None, i=0):
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
    plt.savefig(str(i) + ".png")


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
    i = 0
    for data in loader_correct:
        inputs_cor, labels_cor = data
        print(inputs_cor)
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs_cor)
        imshow(out, title=[l.item() for l in labels_cor], i=i)
        i += 1
    return ()

def training_visualisation(loss_list, acc_list, recall_list):
    """
    :param loss_list: list of epoch loss
    :param acc_list: list of epoch accuracy
    :param recall_list: list of epoch recall
    :return: 2 subplots representing the variation throughout training
    """
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('Training metrics')
    plt.figure(figsize=(16, 6))

    ax1.plot(acc_list, 'r--')
    ax1.plot(recall_list, 'g--')
    ax2.plot(loss_list, 'b-')

    ax1.legend(['Accuracy', 'Recall'])
    ax2.legend(['Loss'])
    ax1.set_ylabel('Metric')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')

    fig.savefig("training_metrics.png")
    plt.show()
