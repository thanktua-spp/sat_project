"""Configuration file for the SATLYT project."""

from pathlib import Path


PROJECT_NAME = "satlyt-project"

id2label = {
    "Aircraft Hangar": 0,
    "Facility": 1,
    "Oil Tanker": 2,
    "Damaged Building": 3,
    "Cement Mixer": 4,
    "Storage Tank": 5,
    "Maritime Vessel": 6,
    "Pylon": 7,
    "Shipping Container": 8,
    "Construction Site": 9,
    }
    
label2id = {v: k for k, v in id2label.items()}
    

# Priority labels for critical detections
CRITICAL_LABELS = {"Facility", "Construction Site", "Damaged Building", "Cement Mixer", "Storage Tank"}
META_FIELDS = ["image_name", "timestamp",
                    "inference_time", "model"]  # always keep these


# Inference Parameters
CONFIDENCE_THRESHOLD = 0.90          # ignore weak detections
SINGLE_INFERENCE = True          # single  or batch image inference
INFERENCE_MODEL_PATH = optim_model_path = Path.cwd() / "models/deployment_model/model_clean_args_fp16.pth"  # path to optimized inference model

# Image and Folder paths    
SINGLE_IMAGE_PATH = Path.cwd() / 'dataset/inference/1038.tif'
INFERENCE_FOLDER_PATH = Path.cwd() / 'dataset/inference/'