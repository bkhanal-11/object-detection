import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg16, resnet50

from datasets import FineTuneDataset, SVMDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = vgg16

# define hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001

#####-----------------FineTuning Model for wrapped images-----------------#####
# load datasets for finetuning and svm classifier
train_dataset_FT = FineTuneDataset(image_path='airplanes/train/images/', annot_path='airplanes/train/annots/')
# test_dataset_FT = FineTuneDataset(image_path='airplanes/test/images/', annot_path='airplanes/test/annots/')

train_loader_FT = torch.utils.data.DataLoader(train_dataset_FT, batch_size=batch_size, shuffle=True)
# test_loader_FT = torch.utils.data.DataLoader(test_dataset_FT, batch_size=batch_size, shuffle=True)

# pre-trained models
if model_name == resnet50:
    model = resnet50(pretrained=True)
    num_features = model.fc.in_features
    fc_layers = nn.Sequential(
                    nn.Linear(num_features, 1),
                    nn.Sigmoid()
                )
    model.fc = fc_layers

    # Set the first few layers of the model to be frozen so that only the last few layers are trainable:
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

if model_name == vgg16:
    model = vgg16(pretrained=True)

    for layers in list(model.children())[:15]:
        for param in layers.parameters():
            param.requires_grad = False

    num_features = model.classifier[-1].in_features
    fc_layers = nn.Sequential(
                    nn.Linear(num_features, 1),
                    nn.Sigmoid()
                )
    model.classifier[-1] = fc_layers
    for param in fc_layers.parameters():
        param.requires_grad = True

# loss function and optimizer
criterion_1 = nn.BCELoss()
optimizer_1 = optim.Adam(model.parameters(), lr=learning_rate)

#train
model.to(device)
model.train(True)
for epoch in range(3):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader_FT):
        images = images.to(device)
        labels = labels.to(device)

        optimizer_1.zero_grad()

        outputs = model(images)
        loss = criterion_1(outputs, labels.float().unsqueeze(1))

        loss.backward()
        optimizer_1.step()

        running_loss += loss.item()
        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels.float().unsqueeze(1)).sum().item()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, 3, i+1, len(train_loader_FT), loss.item()))
    
    epoch_loss = running_loss / len(train_loader_FT)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}\n")
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
train_dataset_SVM = SVMDataset(image_path='airplanes/train/images/', annot_path='airplanes/train/annots/')
# test_dataset_SVM = SVMDataset(image_path='airplanes/test/images/', annot_path='airplanes/test/annots/')

train_loader_SVM = torch.utils.data.DataLoader(train_dataset_SVM, batch_size=batch_size, shuffle=True)
# test_loader_SVM = torch.utils.data.DataLoader(test_dataset_SVM, batch_size=batch_size, shuffle=True)

# Replace last fully connected layer with new linear layer
if model_name == resnet50:
    num_features = model.fc.in_features
    fc_layers = nn.Sequential(
                    nn.Linear(num_features, 2),
                    nn.Softmax()
                )
    model.fc = fc_layers

    # Set the first few layers of the model to be frozen so that only the last few layers are trainable:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

if model_name == vgg16:
    # Freeze all layers except the last fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[-1][0].in_features
    fc_layers = nn.Sequential(
                    nn.Linear(num_features, 2),
                    nn.Softmax()
                )
    model.classifier[-1] = fc_layers
    for param in fc_layers.parameters():
        param.requires_grad = True
# Define optimizer and loss function
optimizer_2 = torch.optim.Adam(model.parameters())
criterion_2 = nn.HingeEmbeddingLoss()

model.to(device)
model.train(True)
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader_SVM):
        images = images.to(device)
        labels = labels.to(device)

        optimizer_2.zero_grad()

        outputs = model(images)
        loss = criterion_2(outputs, labels.float().unsqueeze(1))

        loss.backward()
        optimizer_2.step()

        running_loss += loss.item()
        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels.float().unsqueeze(1)).sum().item()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, len(train_loader_SVM), loss.item()))
    epoch_loss = running_loss / len(train_loader_FT)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}\n")

# # Save model and weights
# torch.save({
#             'model_state_dict': model.state_dict()
#             }, 'my_model_weights.pth')

torch.save(model, 'my_model.pth')
