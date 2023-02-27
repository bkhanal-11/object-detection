import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from datasets import SVMDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define hyperparameters
batch_size = 32
num_epochs = 20
learning_rate = 0.001

model = torch.load('finetuned_vgg16.pth')
model.to(device)

#####-----------------Training Final Model with Linear-SVM-----------------#####
train_dataset_SVM = SVMDataset(image_path='airplanes/train/images/', annot_path='airplanes/train/annots/')

train_loader_SVM = torch.utils.data.DataLoader(train_dataset_SVM, batch_size=batch_size, shuffle=True)

# Replace last fully connected layer with new linear layer
model.classifier[6] = nn.Linear(in_features = 4096, out_features = 2).to(device)
model.classifier[6].requires_grad = True

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

model.classifier[7] = nn.Softmax()

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.HingeEmbeddingLoss()

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader_SVM):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # get the index of the class with the highest score
        _, gt_labels = torch.max(labels.data, 1) 
        total += labels.size(0)
        correct += (predicted == gt_labels).sum().item()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, len(train_loader_SVM), loss.item()))

    epoch_loss = running_loss / len(train_loader_SVM)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}\n")

# Save the trained model
torch.save(model, 'my_model.pth')
