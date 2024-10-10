import torch
import torch.nn as nn

#using this project to create my own first personal nueral network



class NeuralNetwork(nn.Module):
    def __init__(self):
        #intialize nn instance
        super().__init__()
        #3input because images are rgb colors
        #16 different mapped features
        #3x3 kernel is a matrix that extracts edges,textures, etc.
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels = 16, kernel_size = 5)
        # kernel defalt 2x2
        # pool used in reducing image
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels= 16, out_channels = 32, kernel_size = 5)
        #Fully connected layer/applies linear transformations to matrixes
        #used to learn input features
        self.fc1 = nn.Linear(32 *5 *5, 256)
        self.fc2 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 1)

    #function that is responsible for transforming/manipulating matrix inputs
    def forward(self, x):
        pass

