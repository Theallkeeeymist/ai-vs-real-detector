"""
Copy trained models from artifacts to models/ folder for inference.
Run this after training completes.
"""

import os
import shutil
from pathlib import Path

# Find the latest artifact directory
artifact_base = "artifacts"
if not os.path.exists(artifact_base):
    print("❌ No artifacts directory found. Train models first!")
    exit(1)

# Get latest timestamp
timestamps = [d for d in os.listdir(artifact_base) if os.path.isdir(os.path.join(artifact_base, d))]
if not timestamps:
    print("❌ No training artifacts found. Train models first!")
    exit(1)

latest_timestamp = sorted(timestamps)[-1]
trainer_dir = os.path.join(artifact_base, latest_timestamp, "model_trainer")

print(f"Latest training: {latest_timestamp}")
print(f"Copying from: {trainer_dir}")

# Create models directory
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Model mapping
model_files = {
    "model_1_custom_cnn.pth": "model_tiny_vgg_cnn.pth",
    "model_2_efficientnet.pth": "model_1_resnet_b0.pth",
    "model_3_vit.pth": "model_2_ViT_b16.pth",
    "model_4_hybrid.pth": "model_3_Hybrid.pth",
}

# Copy each model
for src_name, dst_name in model_files.items():
    src_path = os.path.join(trainer_dir, src_name)
    dst_path = os.path.join(models_dir, dst_name)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"✅ Copied {src_name} → {dst_name}")
        print(f"   Size: {os.path.getsize(dst_path) / 1024 / 1024:.2f} MB")
    else:
        print(f"❌ NOT FOUND: {src_path}")

print(f"\n✅ All models copied to {models_dir}/")