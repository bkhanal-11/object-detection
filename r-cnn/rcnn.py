import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg16, resnet50

from datasets import FineTuneDataset, SVMDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# load datasets for finetuning and svm classifier
train_dataset_FT = FineTuneDataset(image_path='airplanes/train/images/', annot_path='airplanes/train/annots/')
# test_dataset_FT = FineTuneDataset(image_path='airplanes/test/images/', annot_path='airplanes/test/annots/')

train_loader_FT = torch.utils.data.DataLoader(train_dataset_FT, batch_size=batch_size, shuffle=True)
# test_loader_FT = torch.utils.data.DataLoader(test_dataset_FT, batch_size=batch_size, shuffle=True)

train_dataset_SVM = SVMDataset(image_path='airplanes/train/images/', annot_path='airplanes/train/annots/')
# test_dataset_SVM = SVMDataset(image_path='airplanes/test/images/', annot_path='airplanes/test/annots/')

train_loader_SVM = torch.utils.data.DataLoader(train_dataset_SVM, batch_size=batch_size, shuffle=True)
# test_loader_SVM = torch.utils.data.DataLoader(test_dataset_SVM, batch_size=batch_size, shuffle=True)

# pre-trained models
model = resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

# Set the first few layers of the model to be frozen so that only the last few layers are trainable:
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True
#####-----------------FineTuning Model for wrapped images-----------------#####
# loss function and optimizer
criterion_1 = nn.BCEWithLogitsLoss()
optimizer_1 = optim.Adam(model.fc.parameters(), lr=learning_rate)

#train
model.to(device)
model.train(True)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader_FT):
        images = images.to(device)
        labels = labels.to(device)

        optimizer_1.zero_grad()

        outputs = model(images)
        loss = criterion_1(outputs, labels.float().unsqueeze(1))

        loss.backward()
        optimizer_1.step()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, len(train_loader_FT), loss.item()))
# # test
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         predicted = torch.round(torch.sigmoid(outputs))
#         total += labels.size(0)
#         correct += (predicted == labels.float().unsqueeze(1)).sum().item()

#     print('Test Accuracy: {} %'.format(100 * correct / total))

#####-----------------Training Final Model with Linear-SVM-----------------#####
# Replace last fully connected layer with new linear layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Define optimizer and loss function
optimizer_2 = torch.optim.Adam(model.parameters())
criterion_2 = nn.HingeEmbeddingLoss()

model.train(True)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader_SVM):
        images = images.to(device)
        labels = labels.to(device)

        optimizer_2.zero_grad()

        outputs = model(images)
        loss = criterion_2(outputs, labels.float().unsqueeze(1))

        loss.backward()
        optimizer_2.step()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, len(train_loader_SVM), loss.item()))

# Save model and weights
torch.save({
            'model_state_dict': model.state_dict()
            }, 'my_model_weights.pth')
