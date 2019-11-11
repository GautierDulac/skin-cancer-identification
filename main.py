###Imports
from modelling import validation_model_preconvfeat
from visualizing import final_visualisation, activation_map
from preprocessing import prepare_dsets
from torchvision import models
import torch.optim as optim
import torch
import torch.nn as nn

###Constants
model_select = int(input("Select your model : Enter 1 for vgg , 2 for resnet 18, 3 for mobilenet"))
if model_select == 1:
    model_applied = models.vgg16(pretrained=True)
elif model_select == 2:
    model_applied = models.resnet18(pretrained=True)
else:
    model_applied = models.mobilenet_v2(pretrained=True)

batch_size_train = 64
batch_size_val = 5
batch_size_preconvfeat = 128
shuffle_train = True
shuffle_val = False

num_workers = 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###Main function

for param in model_applied.parameters():
    param.requires_grad = False

if model_select == 2:
    num_ftrs = model_applied.fc.in_features
    model_applied.fc = nn.Linear(num_ftrs, 2)
    optimizer = optim.SGD(model_applied.fc.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

else:
    model_applied.classifier._modules['6'] = nn.Linear(4096, 2)
    model_applied.classifier._modules['7'] = torch.nn.LogSoftmax(dim=1)
    optimizer = torch.optim.Adam(model_applied.classifier.parameters())
    criterion = nn.NLLLoss()

model_applied = model_applied.to(device)
predictions, all_proba, all_classes = validation_model_preconvfeat(model_applied, batch_size_train, batch_size_val,
                                                                   shuffle_train, shuffle_val, num_workers, optimizer, criterion)

print(predictions)
final_visualisation(predictions, all_classes, prepare_dsets())

if model_select == 2:
    activation_map()

# validation_model_preconvfeat(model, batch_size_train, batch_size_val, shuffle_train, shuffle_valid,
# batch_size_preconvfeat, num_workers)
