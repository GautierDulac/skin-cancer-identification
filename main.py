###Imports
from modelling import validation_model_preconvfeat
from torchvision import models
import torch

###Constants
model_vgg = models.vgg16(pretrained=True)
batch_size_train=64
batch_size_val=5
batch_size_preconvfeat = 128
shuffle_train=True
shuffle_val=False
num_workers=6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###Main function

model_vgg = model_vgg.to(device)

predictions, all_proba, all_classes = validation_model_preconvfeat(model_vgg, batch_size_train, batch_size_val, shuffle_train, shuffle_val, batch_size_preconvfeat, num_workers)

print(predictions)