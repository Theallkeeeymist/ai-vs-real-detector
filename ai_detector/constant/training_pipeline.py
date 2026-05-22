import os
import torch
from typing import List

# PIPELINE CONFIG
PIPELINE_NAME = "AIVsRealDetector"
ARTIFACT_DIR = "artifacts"

# DATASET CONFIG
DATASET_DIR = os.getenv("DATASET_DIR", "image_data")
AI_GENERATED_DIR = os.path.join(DATASET_DIR, "image_data/Ai_generated_dataset")
REAL_DATASET_DIR = os.path.join(DATASET_DIR, "image_data/real_dataset")

IMAGE_CATEGORIES = ["animals", "nature", "city", "food", "people"]

DATA_MANIFEST_FILE = "data_manifest.yaml"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 2

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_LABELS = {
    "real_dataset" : 0,
    "ai_generated_dataset" : 1
}

MODELS_DIR = "models"
MODEL_CHECKPOINT_1 = os.path.join(MODELS_DIR, "model_tiny_vgg_cnn.pth")
MODEL_CHECKPOINT_2 = os.path.join(MODELS_DIR, "model_1_resnet_b0.pth")
MODEL_CHECKPOINT_3 = os.path.join(MODELS_DIR, "model_2_ViT_b16.pth")
MODEL_CHECKPOINT_4 = os.path.join(MODELS_DIR, "model_3_Hybrid.pth")

# Model 1: Custom CNN (from your notebook)
CUSTOM_CNN_INPUT = 3
CUSTOM_CNN_HIDDEN = 32
CUSTOM_CNN_OUTPUT = 2

# Model 2: EfficientNet B0
EFFICIENTNET_DROPOUT = 0.3
EFFICIENTNET_HIDDEN = 512

# Model 3: ViT B-16
VIT_OUTPUT_FEATURES = 768

# Model 4: Hybrid
HYBRID_EFFICIENTNET_FEATURES = 1280
HYBRID_VIT_FEATURES = 768
HYBRID_HIDDEN = 512
HYBRID_DROPOUT = 0.4

LEARNING_RATE = 0.001
EPOCHS = 10
DROPOUT_RATE = 0.5
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "ai-vs-real-detector"

METRICS_TO_TRACK = ["accuracy", "precision", "recall", "f1", "roc_auc"]

RANDOM_SEED = 42