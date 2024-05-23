# predict.py

import argparse
import torch
from model import load_checkpoint
from utilities import process_image

def predict(image_path, model, topk=5):
    np_image = process_image(image_path)
    image_tensor = torch.from_numpy(np_image).type(torch.FloatTensor).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
    top_p = top_p.cpu().numpy().tolist()[0]
    top_class = top_class.cpu().numpy().tolist()[0]
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class]
    return top_p, top_class

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image along with the probability of that name")
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.input, model, topk=args.top_k)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(cls)] for cls in classes]

    print("Predicted Classes and Probabilities:")
    for cls, prob in zip(classes, probs):
        print(f"{cls}: {prob:.4f}")

if __name__ == '__main__':
    main()
