###Imports
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from PIL import ImageFile
import json

###Constants

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Using gpu: %s ' % torch.cuda.is_available())

###Main function


###Core functions
model_vgg = models.vgg16(pretrained=True)


fpath = 'data/imagenet_class_index.json'

with open(fpath) as f:
    class_dict = json.load(f)
dic_imagenet = [class_dict[str(i)][1] for i in range(len(class_dict))]

inputs_try, labels_try = inputs_try.to(device), labels_try.to(device)

model_vgg = model_vgg.to(device)

outputs_try = model_vgg(inputs_try)

m_softm = nn.Softmax(dim=1)
probs = m_softm(outputs_try)
vals_try, preds_try = torch.max(probs, dim=1)

torch.sum(probs, 1)

out = torchvision.utils.make_grid(inputs_try.data.cpu())

imshow(out, title=[dset_classes[x] for x in labels_try.data.cpu()])

"""### Modifying the last layer and setting the gradient false to all layers"""

print(model_vgg)

"""We'll learn about what these different blocks do later in the course. For now, it's enough to know that:

- Convolution layers are for finding small to medium size patterns in images -- analyzing the images locally
- Dense (fully connected) layers are for combining patterns across an image -- analyzing the images globally
- Pooling layers downsample -- in order to reduce image size and to improve invariance of learned features

In this practical example, our goal is to use the already trained model and just change the number of output classes. To this end we replace the last ```nn.Linear``` layer trained for 1000 classes to ones with 2 classes. In order to freeze the weights of the other layers during training, we set the field ```required_grad=False```. In this manner no gradient will be computed for them during backprop and hence no update in the weights. Only the weights for the 2 class layer will be updated.
"""

for param in model_vgg.parameters():
    param.requires_grad = False
model_vgg.classifier._modules['6'] = nn.Linear(4096, 2)
model_vgg.classifier._modules['7'] = torch.nn.LogSoftmax(dim=1)

print(model_vgg.classifier)

model_vgg = model_vgg.to(device)

"""## Training fully connected module

### Creating loss function and optimizer

"""

criterion = nn.NLLLoss()
lr = 0.001
optimizer_vgg = torch.optim.SGD(model_vgg.classifier[6].parameters(), lr=lr)

"""### Training the model"""


def train_model(model, dataloader, size, epochs=1, optimizer=None):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
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
        epoch_loss = running_loss / size
        epoch_acc = running_corrects.data.item() / size
        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))


train_model(model_vgg,loader_train,size=dset_sizes['train'],epochs=2,optimizer=optimizer_vgg)

def test_model(model, dataloader, size):
    model.eval()
    predictions = np.zeros(size)
    all_classes = np.zeros(size)
    all_proba = np.zeros((size, 2))
    i = 0
    running_loss = 0.0
    running_corrects = 0
    for inputs, classes in dataloader:
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, classes)
        _, preds = torch.max(outputs.data, 1)
        # statistics
        running_loss += loss.data.item()
        running_corrects += torch.sum(preds == classes.data)
        predictions[i:i + len(classes)] = preds.to('cpu').numpy()
        all_classes[i:i + len(classes)] = classes.to('cpu').numpy()
        all_proba[i:i + len(classes), :] = outputs.data.to('cpu').numpy()
        i += len(classes)
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.data.item() / size
    print('Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))
    return predictions, all_proba, all_classes


predictions, all_proba, all_classes = test_model(model_vgg, loader_valid, size=dset_sizes['valid'])

# Get a batch of training data
inputs, classes = next(iter(loader_valid))

out = torchvision.utils.make_grid(inputs[0:n_images])

imshow(out, title=[dset_classes[x] for x in classes[0:n_images]])

outputs = model_vgg(inputs[:n_images].to(device))
print(torch.exp(outputs))

classes[:n_images]

"""## Speeding up the learning by precomputing features"""

x_try = model_vgg.features(inputs_try)

def preconvfeat(dataloader):
    conv_features = []
    labels_list = []
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        x = model_vgg.features(inputs)
        conv_features.extend(x.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
    conv_features = np.concatenate([[feat] for feat in conv_features])
    return (conv_features, labels_list)


conv_feat_train,labels_train = preconvfeat(loader_train)

conv_feat_valid,labels_valid = preconvfeat(loader_valid)

"""### Creating a new data generator

We will not load images anymore, so we need to build our own data loader.
"""

dtype = torch.float
train_dataset = [[torch.from_numpy(f).type(dtype), torch.tensor(l).type(torch.long)] for (f, l) in
                 zip(conv_feat_train, labels_train)]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)


def train_model(model, size, epochs=1, optimizer=None, dataloader=None):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = int(0)
        for inputs, classes in dataloader:
            inputs = inputs.to(device)
            classes = classes.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs, classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs.data, 1)
            # statistics
            running_loss += loss.data.item()
            running_corrects += torch.sum(preds == classes.data)
        epoch_loss = running_loss / size
        epoch_acc = running_corrects.item() / size
        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))


train_model(model_vgg.classifier,size=dset_sizes['train'],epochs=50,optimizer=optimizer_vgg,dataloader=train_loader)

valid_dataset = [[torch.from_numpy(f).type(dtype), torch.tensor(l).type(torch.long)] for (f, l) in
                 zip(conv_feat_valid, labels_valid)]
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)


def test_model(model, size, dataloader=None):
    model.eval()
    predictions = np.zeros(size)
    all_classes = np.zeros(size)
    all_proba = np.zeros((size, 2))
    i = 0
    running_loss = 0.0
    running_corrects = 0
    for inputs, classes in dataloader:
        inputs = inputs.to(device)
        classes = classes.to(device)
        inputs = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        loss = criterion(outputs, classes)
        _, preds = torch.max(outputs.data, 1)
        # statistics
        running_loss += loss.data.item()
        running_corrects += torch.sum(preds == classes.data)
        # print(i)
        predictions[i:i + len(classes)] = preds.to('cpu').numpy()
        all_classes[i:i + len(classes)] = classes.to('cpu').numpy()
        all_proba[i:i + len(classes), :] = outputs.data.to('cpu').numpy()
        # print(preds.cpu())
        i += len(classes)
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.data.item() / size
    print('Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))
    return predictions, all_proba, all_classes


predictions, all_proba, all_classes = test_model(model_vgg.classifier, size=dset_sizes['valid'],
                                                 dataloader=valid_loader)
