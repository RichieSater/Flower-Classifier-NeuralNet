# model.py

import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict

def build_model(arch='vgg16', hidden_units=512):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        raise ValueError("Unsupported architecture")
    
    for param in model.features.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 1024)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(1024, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return model

def train_model(model, dataloaders, criterion, optimizer, epochs, device):
    steps = 0
    running_loss = 0
    print_every = 10

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        validation_loss += loss.item()

                        # Calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

                running_loss = 0
                model.train()

def save_checkpoint(model, optimizer, epochs, class_to_idx, save_path='checkpoint.pth'):
    checkpoint = {
        'epochs': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx,
        'classifier': model.classifier
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, 1024)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(1024, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
