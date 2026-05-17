"""
Script to create reproducible split metadata from your raw dataset structure.
Run this ONCE and commit split_metadata.yaml to git.
"""

import os
import yaml
import random
from typing import Dict, List
from pathlib import Path

def create_split_metadata(
    dataset_dir: str,
    output_file: str = "data_manifest.yaml",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> None:
    """
    Create reproducible split metadata from your folder structure.
    
    Example:
        /home/allkeeey/datasets/ai_vs_real/
        ├── ai_generated_dataset/
        │   ├── animals/
        │   ├── nature/
        │   ...
        └── real_dataset/
            ├── animals/
            ├── nature/
            ...
    
    Output:
        data_manifest.yaml with structure:
        train:
          real_dataset:
            animals: [img1.jpg, img2.jpg, ...]
          ai_generated_dataset:
            ...
        val:
          ...
        test:
          ...
    """
    
    random.seed(seed)
    
    manifest = {
        "train": {},
        "val": {},
        "test": {}
    }
    
    # For each class folder (real_dataset, ai_generated_dataset)
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        # Initialize nested structure
        for split in ["train", "val", "test"]:
            manifest[split][class_name] = {}
        
        # For each category (animals, nature, city, food, people)
        for category in os.listdir(class_path):
            category_path = os.path.join(class_path, category)
            
            if not os.path.isdir(category_path):
                continue
            
            # Get all images in this category
            images = [f for f in os.listdir(category_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Shuffle and split
            random.shuffle(images)
            n = len(images)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_images = images[:train_end]
            val_images = images[train_end:val_end]
            test_images = images[val_end:]
            
            # Store in manifest
            manifest["train"][class_name][category] = train_images
            manifest["val"][class_name][category] = val_images
            manifest["test"][class_name][category] = test_images
            
            print(f"✓ {class_name}/{category}: "
                  f"train={len(train_images)}, "
                  f"val={len(val_images)}, "
                  f"test={len(test_images)}")
    
    # Save to YAML
    with open(output_file, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False)
    
    print(f"\n✅ Manifest saved to {output_file}")
    
    # Print summary
    total_train = sum(
        len(imgs) 
        for class_dict in manifest["train"].values()
        for imgs in class_dict.values()
    )
    total_val = sum(
        len(imgs)
        for class_dict in manifest["val"].values()
        for imgs in class_dict.values()
    )
    total_test = sum(
        len(imgs)
        for class_dict in manifest["test"].values()
        for imgs in class_dict.values()
    )
    
    print(f"\n📊 Split Summary:")
    print(f"  Train: {total_train} images")
    print(f"  Val:   {total_val} images")
    print(f"  Test:  {total_test} images")
    print(f"  Total: {total_train + total_val + total_test} images")


if __name__ == "__main__":
    # Run this once from project root
    from ai_detector.constant.training_pipeline import (
        DATASET_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, DATA_MANIFEST_FILE
    )
    
    create_split_metadata(
        dataset_dir=DATASET_DIR,
        output_file=DATA_MANIFEST_FILE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )