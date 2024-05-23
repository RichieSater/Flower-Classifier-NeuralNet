# Imports here

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# TODO Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
}

# TODO Using the image datasets and the transforms, define the dataloaders
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64),
}

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Load a pre-trained network - will use Vgg16
model = models.vgg16(pretrained=True)

# Freeze the feature extraction layers
for param in model.features.parameters():
    param.requires_grad = False

# Define a new, untrained feed-forward network as a classifier
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

# Define the criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Move the model to the GPU - I have a 4090RTX
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training the classifier - will use backpropagation 
epochs = 5
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

                    # Calculate accuracy - to help determine best hyperparameters
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

            # TODO: Do validation on the test set

# Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
test_loss = 0
accuracy = 0

with torch.no_grad():  # Turn off gradients for testing
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        # Calculate accuracy
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

# Calculate the average test loss and accuracy
test_loss = test_loss / len(dataloaders['test'])
accuracy = accuracy / len(dataloaders['test'])

print(f"Test Loss: {test_loss:.3f}.. "
      f"Test Accuracy: {accuracy:.3f}")

# TODO: Save the checkpoint 

# Attach the class-to-index mapping to the model
model.class_to_idx = image_datasets['train'].class_to_idx

# Define a checkpoint dictionary
checkpoint = {
    'epochs': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_to_idx': model.class_to_idx,
    'classifier': model.classifier
}

# Save the checkpoint
torch.save(checkpoint, 'flower_classifier_checkpoint.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    # Load a pre-trained network
    model = models.vgg16(pretrained=True)
    
    # Freeze the feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Rebuild the classifier
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
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the class to index mapping
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Load the optimizer state
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

# Example usage:
filepath = 'flower_classifier_checkpoint.pth'
model, optimizer = load_checkpoint(filepath)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Resize the image
    aspect_ratio = image.width / image.height
    if image.width < image.height:
        image = image.resize((256, int(256 / aspect_ratio)))
    else:
        image = image.resize((int(256 * aspect_ratio), 256))
    
    # Center crop the image
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    
    # Convert image to numpy array
    np_image = np.array(image) / 255.0
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions to match PyTorch's expected input
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Process the image
    np_image = process_image(image_path)
    
    # Convert the image to a PyTorch tensor
    image_tensor = torch.from_numpy(np_image).type(torch.FloatTensor)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Set model to evaluation mode and turn off gradients
    model.eval()
    with torch.no_grad():
        # Get the model output
        output = model(image_tensor)
        ps = torch.exp(output)
        
        # Get the top K probabilities and corresponding classes
        top_p, top_class = ps.topk(topk, dim=1)
        
    # Convert probabilities and classes to lists
    top_p = top_p.cpu().numpy().tolist()[0]
    top_class = top_class.cpu().numpy().tolist()[0]
    
    # Map the class indices to the actual class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class]
    
    return top_p, top_class

# TODO: Display an image along with the top 5 classes

def sanity_check(image_path, model, cat_to_name, topk=5):
    # Make prediction
    probs, classes = predict(image_path, model, topk)
    
    # Map classes to flower names
    flower_names = [cat_to_name[str(cls)] for cls in classes]
    
    # Display the image
    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)
    
    img = process_image(image_path)
    img = torch.from_numpy(img)
    imshow(img, ax=ax, title=flower_names[0])
    
    # Display the bar chart
    plt.subplot(2, 1, 2)
    y_pos = np.arange(len(flower_names))
    plt.barh(y_pos, probs, align='center')
    plt.yticks(y_pos, flower_names)
    plt.xlabel('Probability')
    plt.gca().invert_yaxis()
    
    plt.show()

# Example usage
image_path = 'assets/Flowers.png'
sanity_check(image_path, model, cat_to_name)