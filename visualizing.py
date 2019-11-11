###Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from numpy.random import permutation
import cv2
from torch.nn import functional as F

import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
import pdb
from matplotlib.pyplot import imshow

###Constants
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    plt.imsave("Corrects-" + str(i) + "-" + str(title) + ".png", inp)


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
    idx = permutation(correct)[:n_view]
    loader_correct = torch.utils.data.DataLoader([dsets['valid'][x] for x in idx], batch_size=n_view, shuffle=True)
    i = 0
    for data in loader_correct:
        inputs_cor, labels_cor = data
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs_cor)
        imshow(out, title=[l.item() for l in labels_cor], i=i)
        i += 1
    return ()


def activation_map(resnet_model, predictions, all_classes, dsets, img_number="2"):
    """

    :param predictions:
    :param all_classes:
    :param dsets:
    :return:
    """

    #IMG_URL = 'https://www.dropbox.com/s/pizj50193hzzsmp/2.jpg?dl=0'
    IMG_URL = 'data/train/malignant/2.jpg'

    # Activation map part
    finalconv_name = 'layer4'
    resnet_model.eval()

    # hook the feature extractor
    # see https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html
    # for more explanations
    features_blobs = []

    def hook_feature(module, input, output):
        print('Inside ' + module.__class__.__name__ + ' forward')
        print('')
        print('input: ', type(input))
        print('input[0]: ', type(input[0]))
        print('output: ', type(output))
        print('')
        print('input size:', input[0].size())
        print('output size:', output.data.size())
        features_blobs.append(output.data.cpu().numpy())

    resnet_model._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(resnet_model.parameters())
    weight_softmax = np.squeeze(params[-2].to('cpu').data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    #response = requests.get(IMG_URL)
    #print(response.content)
    img_pil = Image.open(IMG_URL)
    img_pil.save('test.jpg')

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).to(device)
    logit = resnet_model(img_variable)

    classes = {0: "benign", 1: "malignant"}

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.to('cpu').numpy()
    idx = idx.to('cpu').numpy()

    # output the prediction
    for i in range(0, 2):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
    img = cv2.imread('test.jpg')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('CAM_True_' + classes[idx[i]] + '.jpg', result)

    # generate class activation mapping for the top2 prediction
    CAM1s = returnCAM(features_blobs[0], weight_softmax, [idx[2]])

    # render the CAM and output
    print('output CAM.jpg for the top2 prediction: %s' % classes[idx[2]])
    img = cv2.imread('test.jpg')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAM1s[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('CAM_False_' + classes[idx[i]] + '.jpg', result)
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
