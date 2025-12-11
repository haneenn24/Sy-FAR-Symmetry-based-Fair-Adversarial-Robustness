import os
import shutil
import random
from typing import List


def split_dataset(
    dataset_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> None:
    """
    Split a dataset into train/val/test subsets in ImageFolder format.

    Each class in `dataset_dir` should be a subdirectory containing images.
    Output structure:

        output_dir/
            train/<class_name>/*.jpg
            val/<class_name>/*.jpg
            test/<class_name>/*.jpg

    Args:
        dataset_dir (str): Path to root dataset directory containing class subfolders.
        output_dir (str): Path to write the split dataset.
        train_ratio (float): Fraction of images assigned to the training set.
        val_ratio (float): Fraction of images assigned to the validation set.
        test_ratio (float): Fraction of images assigned to the test set.

    Raises:
        AssertionError: If the provided ratios do not sum to 1.0.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1."

    # Create split directories
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Process each class directory
    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # Skip non-directory entries

        images: List[str] = [
            fname for fname in os.listdir(class_path)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        split_mapping = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        # Write images to split directories
        for split, img_list in split_mapping.items():
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img_name in img_list:
                src = os.path.join(class_path, img_name)
                dst = os.path.join(split_class_dir, img_name)
                shutil.copy2(src, dst)

        print(f"Processed class '{class_name}': {total} images.")

    print(
        f"\nDataset split complete:\n"
        f"  Train: {train_ratio*100:.1f}%\n"
        f"  Val:   {val_ratio*100:.1f}%\n"
        f"  Test:  {test_ratio*100:.1f}%"
    )


# Example usage
if __name__ == "__main__":
    dataset_dir = "/path_to_datasets/"  # Replace with your path
    output_dir = "/path_to_datasets_splitted"
    split_dataset(dataset_dir, output_dir)
