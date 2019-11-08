###Imports
from modelling import validation_model_preconvfeat
from visualizing import final_visualisation
from preprocessing import prepare_dsets
from torchvision import models
import torch
import torch.nn as nn

###Constants
model_select = int(input("Select your model : Enter 1 for vgg , 2 for resnet 18, 3 for resnet 50"))
if model_select == 1:
    model_applied = models.vgg16(pretrained=True)
elif model_select == 2:
    model_applied = models.resnet18(pretrained=True)
else:
    model_applied = models.resnext50_32x4d(pretrained=True)


batch_size_train=64
batch_size_val=5
batch_size_preconvfeat = 128
shuffle_train=True
shuffle_val=False

num_workers=6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



###Main function

for param in model_applied.parameters():
    param.requires_grad = False

if model_select == 1:
    model_applied.classifier._modules['6'] = nn.Linear(4096, 2)
    model_applied.classifier._modules['7'] = torch.nn.LogSoftmax(dim = 1)
    model_applied = model_applied.to(device)
    predictions, all_proba, all_classes = validation_model_preconvfeat(model_applied, batch_size_train, batch_size_val,
                                                                       shuffle_train, shuffle_val,
                                                                       batch_size_preconvfeat, num_workers)
    print(predictions)
    final_visualisation(predictions, all_classes, prepare_dsets())


elif model_select == 2:
    num_ftrs = model_conv.fc.in_features
    model_applied.fc = nn.Linear(num_ftrs, 2)
    model_applied = model_applied.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_applied = optim.SGD(model_applied.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_applied, step_size=7, gamma=0.1)
    model_applied = train_model(model_applied, criterion, optimizer_conv,exp_lr_scheduler, num_epochs=25)



