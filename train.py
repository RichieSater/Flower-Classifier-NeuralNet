import argparse
import torch
from torch import nn, optim
from model import build_model, train_model, save_checkpoint
from utilities import load_data

def main():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint")
    parser.add_argument('data_directory', type=str, help='Path to the folder of flower images')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Path to save the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg13 or vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()
    
    print("Loading data...")
    dataloaders, class_to_idx = load_data(args.data_directory)
    print("Data loaded successfully.")
    
    print(f"Building model with architecture {args.arch}...")
    model = build_model(arch=args.arch, hidden_units=args.hidden_units)
    print(f"Model built successfully with {args.hidden_units} hidden units.")
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        
    model.to(device)

    print(f"Starting training for {args.epochs} epochs...")
    train_model(model, dataloaders, criterion, optimizer, args.epochs, device)
    print("Training completed.")

    print(f"Saving checkpoint to {args.save_dir}...")
    save_checkpoint(model, optimizer, args.epochs, class_to_idx, args.save_dir)
    print("Checkpoint saved successfully.")

if __name__ == '__main__':
    main()
