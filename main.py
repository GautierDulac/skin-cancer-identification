###Imports
from modelling import validation_model_preconvfeat
from visualizing import final_visualisation, activation_map
from preprocessing import prepare_dsets
from model_small_cancerid import validation_model_preconvfeat_2
from torchvision import models
import torch.optim as optim
import torch
import torch.nn as nn


###Constants
model_select = int(input("Select your model : Enter 1 for vgg , 2 for resnet 18, 3 for mobilenet, 4 for custom built small net\n"))
num_epochs = int(input("Select epochs number\n"))
lr = float(input("Select learning rate\n"))

if (not (model_select in [1, 2, 3, 4])) or num_epochs <= 0 or lr <= 0:
    raise ValueError("Paramètres incorrects")


if model_select == 1:
    model_applied = models.vgg16(pretrained=True)
elif model_select == 2:
    model_applied = models.resnet18(pretrained=True)
elif model_select == 3:
    model_applied = models.mobilenet_v2(pretrained=True)

batch_size_train = 64
batch_size_val = 5
batch_size_preconvfeat = 128
shuffle_train = True
shuffle_val = False

num_workers = 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

###Main function

for param in model_applied.parameters():
    param.requires_grad = False


if model_select == 1 :
    model_applied.classifier._modules['6'] = nn.Linear(4096, 2)
    model_applied.classifier._modules['7'] = torch.nn.LogSoftmax(dim=1)
    optimizer = torch.optim.Adam(model_applied.classifier.parameters())
    criterion = nn.NLLLoss()

elif model_select == 2:
    num_ftrs = model_applied.fc.in_features
    model_applied.fc = nn.Linear(num_ftrs, 2)
    optimizer = optim.SGD(model_applied.fc.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

elif model_select == 3:
    model_applied.classifier._modules['1'] = nn.Linear(1280, 2)
    model_applied.classifier._modules['2'] = torch.nn.LogSoftmax(dim=1)
    optimizer = torch.optim.Adam(model_applied.classifier.parameters())
    criterion = nn.NLLLoss()
    
elif model_select == 4:
    from model_small_cancerid import classifier
    from model_small_cancerid import validation_model_preconvfeat
    classif = classifier()
    model_applied = classif.cuda()
    optimizer = torch.optim.Adam(model_applied.parameters(), lr = 0.001)
    
    

model_applied = model_applied.to(device)

if model_select == 4:
    predictions, all_proba, all_classes = validation_model_preconvfeat2(model_applied, batch_size_train, batch_size_val,
                                                                   shuffle_train, shuffle_val, num_workers, optimizer, criterion, model_select, num_epochs, lr)
else:
    predictions, all_proba, all_classes = validation_model_preconvfeat(model_applied, batch_size_train, batch_size_val,
                                                                   shuffle_train, shuffle_val, num_workers, optimizer, criterion, model_select, num_epochs, lr)

print(predictions)
final_visualisation(predictions, all_classes, prepare_dsets())

if model_select == 2:
    file_name = str(input('Please Select the file on which to save the activation map : Enter the path from the project directory\n'))
    save_name = str(input('Save the image to the following name\n'))

    activation_map(model_applied, file_name, save_name)

final_visualisation(predctions, all_classes, prepare_dsets())
