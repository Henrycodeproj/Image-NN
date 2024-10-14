import torch
import torch.nn as nn
import torchvision

#using this project to create my own first personal nueral network

def transform_dataset():
    transform = torchvision.transforms.Compose([ 
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.ImageFolder(root='data/Train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    return train_loader



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
        
        self.fc_input_size = self._get_conv_output_size((3, 224, 224))

        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 1)

    def _get_conv_output_size(self, shape):
        """ Helper function to calculate the size after conv and pooling layers """
        x = torch.rand(1, *shape)  # Create a dummy input tensor
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x.numel() 

    #function that is responsible for transforming/manipulating matrix inputs
    def forward(self, x):
        #sets all negative numbers in the vectors to 0
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

model = NeuralNetwork()

trainloader = transform_dataset()

# 3. Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 4. Training loop
for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')