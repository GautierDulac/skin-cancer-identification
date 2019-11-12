### Imports
import torch.nn as nn
import torch.nn.functional as F

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
        size_features = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size_features:
            num_features *= s
        return num_features



