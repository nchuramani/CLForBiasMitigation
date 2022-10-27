import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_features=24, num_of_labels=12):
        super(Net, self).__init__()
        self.num_features = num_features
        self.num_of_labels = 12
        self.layer1 = self.one_block(1, 3)
        self.layer2 = self.one_block(2, self.num_features)
        self.layer3 = self.one_block(4, 2*self.num_features)
        self.layer4 = self.one_block(8, 4*self.num_features)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(768, 8 * self.num_features)
        self.fc2 = nn.Linear(8 * self.num_features, 4 * self.num_features)
        self.fc3 = nn.Linear(4 * self.num_features, 2 * self.num_features)
        self.last2 = nn.Linear(2 * self.num_features, 12)
        self.last = self.last2
    def forward(self, x):
        out2 = self.layer1(x)
        out2 = self.layer2(out2)
        out2 = self.layer3(out2)
        out2 = self.layer4(out2)
        out2 = self.flat(out2)
        out2 = F.relu(self.fc1(out2))
        out2 = nn.Dropout2d(0.3)(out2)
        out2 = F.relu(self.fc2(out2))
        out2 = F.relu(self.fc3(out2))
        out2 = self.last2(out2)
        out2 = nn.Sigmoid()(out2)
        out = {}
        out["All"] = out2
        return out
    
    def one_block(self, a, inp):
        block = nn.Sequential(
            nn.Conv2d(inp, a * self.num_features, kernel_size=(3,3), padding=1), #padding=same
            nn.ReLU(),
            nn.BatchNorm2d(a * self.num_features),
            nn.Conv2d(a * self.num_features, a * self.num_features, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(a * self.num_features),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout(p=0.4)
                ) 
        return block
    
def simple_cnn():
    return Net()
