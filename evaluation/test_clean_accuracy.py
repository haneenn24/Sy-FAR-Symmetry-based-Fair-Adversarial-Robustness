# evaluation/test_clean_accuracy.py

"""
Evaluate clean (benign) accuracy of a trained model.

Usage:
    python evaluation/test_clean_accuracy.py \
        --model-path ./glass/donemodel/model_name.pt \
        --batch-size 64 \
        --data-dir ./data
"""

import argparse
import copy
import torch
import torch.nn as nn
from utils import data_process
from models.vgg16 import VGG_16


def evaluate_clean_accuracy(model: nn.Module, dataloaders: dict, dataset_sizes: dict) -> float:
    """
    Evaluate clean accuracy of the model on the test set.

    Args:
        model (nn.Module): Trained model.
        dataloaders (dict): Dataloaders for train/val/test.
        dataset_sizes (dict): Sizes of datasets.

    Returns:
        float: Test accuracy in percentage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = copy.deepcopy(model).to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloaders["test"]:
            # convert RGB → BGR (dataset-specific requirement)
            images = images[:, [2, 1, 0], :, :].to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Accuracy on {total} test images: {accuracy:.2f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate clean accuracy of a trained model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for testing")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to dataset")
    args = parser.parse_args()

    # Load model
    model = VGG_16()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    print(f"Loaded model from {args.model_path}")

    # Load data
    dataloaders, dataset_sizes, class_names = data_process(batch_size=args.batch_size, data_dir=args.data_dir)

    # Evaluate
    evaluate_clean_accuracy(model, dataloaders, dataset_sizes)


if __name__ == "__main__":
    main()
