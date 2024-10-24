import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

# 1. Augmentations for training data (with data augmentation)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Test/Validation transform
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 2. Load the dataset 
full_dataset = datasets.ImageFolder(root='Data/Train', transform=train_transform)

# 3. Split dataset into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Use test_transform for validation set to avoid augmentation
val_dataset.dataset.transform = test_transform

# 4. DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. Define the CNN model for binary classification (same as before)
class BananaClassifier(nn.Module):
    def __init__(self):
        super(BananaClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    
    # Relu used to fix vanishing gradient problem
    # function that manipulates matrix/tensor data
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) #checks any the matrix/tensors for negative numbers and changes them to 0.
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 6. Initialize the model, loss function, and optimizer
model = BananaClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training and Validation Loop
num_epochs = 15
for epoch in range(num_epochs):
    # Training
    model.train()
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        labels = labels.float().view(-1, 1)  # Reshape labels
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    # Validation
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = labels.float().view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            # Calculate validation accuracy
            predicted = (outputs > 0.5).float()  # Apply threshold
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print model accuracy against non trained data
    train_loss = running_train_loss / len(train_loader)
    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# 8. Save the Model 
torch.save(model.state_dict(), 'banana_classifier.pth')
print("Model saved as 'banana_classifier.pth'")

# 9. Load the Model for Testing 
model.load_state_dict(torch.load('banana_classifier.pth'))
model.eval()  # Set the model to evaluation mode

# 10. Function to Test a Single Image 
def test_single_image(image_path, model):
    image = Image.open(image_path)
    image = test_transform(image).unsqueeze(0)  # Apply test transformations
    with torch.no_grad():
        output = model(image)
        prediction = (output > 0.5).float()
        print(f"Raw model output: {output.item()}")
        if prediction.item() == 1:
            print("This is a banana!")
        else:
            print("This is not a banana.")

# Test an image
test_image_path = "pixels.jpg" 
test_single_image(test_image_path, model)
