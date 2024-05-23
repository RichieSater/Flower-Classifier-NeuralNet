import argparse
import json
import torch
from model import load_checkpoint
from utilities import process_image, predict_image

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image with a trained model")
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category to name mapping JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    print("Loading model checkpoint...")
    model, _ = load_checkpoint(args.checkpoint)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on device: {device}")

    print("Processing image...")
    image = process_image(args.input)
    
    print("Predicting class...")
    probs, classes = predict_image(image, model, topk=args.top_k, device=device)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        print("Category to name mapping loaded successfully.")
        print(cat_to_name)
        class_names = [cat_to_name[str(cls)] for cls in classes]
    else:
        class_names = classes

    print("Prediction results:")
    for prob, cls in zip(probs, class_names):
        print(f"{cls}: {prob:.4f}")

if __name__ == '__main__':
    main()
