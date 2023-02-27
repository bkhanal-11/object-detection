import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16

from datasets import FineTuneDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = vgg16

# define hyperparameters
batch_size = 64
num_epochs = 10
learning_rate = 0.001

#####-----------------FineTuning Model for wrapped images-----------------#####
# load datasets for finetuning
train_dataset_FT = FineTuneDataset(image_path='airplanes/train/images/', annot_path='airplanes/train/annots/')

train_loader_FT = torch.utils.data.DataLoader(train_dataset_FT, batch_size=batch_size, shuffle=True)

model = vgg16(weights='VGG16_Weights.DEFAULT')

# Replace the last fully connected layer
model.classifier[-1] = nn.Linear(in_features=4096, out_features=1)
model.classifier.add_module('sigmoid', nn.Sigmoid())

# Freeze all layers except the last two fully connected layers
for name, param in model.named_parameters():
    if 'classifier.3' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model.to(device)
# summary(model, (3, 224, 224))

# loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader_FT):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels.float().unsqueeze(1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels.float().unsqueeze(1)).sum().item()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, len(train_loader_FT), loss.item()))
    
    epoch_loss = running_loss / len(train_loader_FT)
    epoch_acc = correct / total
    print(f"\nEpoch {epoch+1} ---> loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}\n")

torch.save(model, 'finetuned_vgg16.pth')
