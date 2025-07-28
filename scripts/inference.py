from rfdetr import RFDETRNano
import supervision as sv
from pathlib import Path

import os
import json
import cv2
import time
from datetime import datetime

from config import (
    label2id,
    CONFIDENCE_THRESHOLD,
    META_FIELDS,
    CRITICAL_LABELS,
    SINGLE_INFERENCE,
    INFERENCE_MODEL_PATH,
    SINGLE_IMAGE_PATH,
    INFERENCE_FOLDER_PATH,
)


def single_sahi_inference(image_path, model, threshold):
    """Single image SAHI inference"""
    
    inference_start = time.time()
    
    # Load and resize image
    image = cv2.imread(str(image_path))
        
    # Define Sahi callback
    def callback(image):
        return model.predict(image, threshold=threshold)
    
    # Create slicer and run inference
    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(540, 540),  # for improved performance
        overlap_ratio_wh=None,
        overlap_wh=(32, 32),  # 25% overlap
        iou_threshold=0.5
        )
    detections = slicer(image)
    inference_time = time.time() - inference_start
        
    print(f"Single inference on {Path(image_path).name}:")
    print(f"  Inference time: {inference_time:.3f}s")
    
    return image, detections, inference_time
    
    
def batch_sahi_inference(folder_path, model, threshold):
    """Batch SAHI inference on all images in folder"""
        
    # Get all image files
    folder_path = Path(folder_path)
    image_files = list(folder_path.glob('*.tif'))
    
    if len(image_files) < 1:
        print(f"No images found in {folder_path}. Please check the path.")
        return []
    
    print(f"Found {len(image_files)} images")
    
    # Define callback
    def callback(image):
        return model.predict(image, threshold=threshold)
    
    # Create slicer
    slicer = sv.InferenceSlicer(callback=callback)
    
    # Process all images
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
        image = cv2.imread(str(image_path))
        
        # Run SAHI inference
        inference_start = time.time()
        detections = slicer(image)
        inference_time = time.time() - inference_start
        
        # Store results
        results.append((image_path, image, detections, inference_time))
        
        print(f"  Found {len(detections)} objects in {inference_time:.3f}s")
    
    print("\nBatch inference completed:")
    return results


def annotate_and_display(image, detections, label2id, image_name=""):
    """Helper function to annotate and display results"""
    
    labels = [
        f"{label2id[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    
    if image_name:
        print(f"\nResults for {image_name}:")
    
    sv.plot_image(annotated_image)
    return annotated_image


def format_detections(
    detections,
    threshold=0.5,
    ):
    """Format detections into a structured list"""
    
    boxes = detections.xyxy
    scores = detections.confidence
    class_ids = detections.class_id

    # Filter indices where score >= threshold
    keep_indices = scores >= threshold

    return [
        {
            "bbox": box.tolist(),
            "score": float(score),
            "label": label2id[int(class_id)],
        }
        for box, score, class_id in zip(boxes[keep_indices], scores[keep_indices], class_ids[keep_indices])
    ]


def run_inference(image_path=None, 
                  folder_path=None,
                  model_path=None, 
                  threshold=0.5,
                  single_infernce=True):
    """Run single or batch inference based on input and returns metadata"""

    metadata_list = []
    timestamp = datetime.utcnow().isoformat()
    
    # Load selected model
    inference_model = RFDETRNano(pretrain_weights=str(model_path))
    
    if single_infernce: # Single image inference
        image, detections, inference_time = single_sahi_inference(image_path, inference_model, threshold=threshold)
        metadata = {
            "timestamp": timestamp,
            "image_name": Path(image_path).name,
            "image_size_kb": os.path.getsize(image_path) // 1024,
            "detections": format_detections(detections),
            "inference_time": inference_time,
            "model": model_path.name,
        }
        metadata_list.append(metadata)
        annotate_and_display(image, detections, label2id, image_path.name)
        
    else: # Batch inference
        batch_results = batch_sahi_inference(folder_path, inference_model, threshold=threshold)
        # Display all batch results
        for image_path, image, detections, inference_time in batch_results:
            metadata = {
                "timestamp": timestamp,
                "image_name": Path(image_path).name,
                "image_size_kb": os.path.getsize(image_path) // 1024,
                "detections": format_detections(detections),
                "inference_time": inference_time,
                "model": model_path.name,
            }
            metadata_list.append(metadata)
            annotate_and_display(image, detections, label2id, image_path.name)
            
    return json.dumps(metadata_list)


def filter_metadata(raw_json, threshold):
    """
    Takes the raw inference‑metadata JSON (string or list[dict]),
    returns a tuple:  (is_critical, filtered_metadata_dict)
    """
    data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    assert isinstance(data, list) and data, "Expect list[dict] from inference"

    meta = {k: data[0][k] for k in META_FIELDS}        # copy basics
    critical_dets = [
        det for det in data[0]["detections"]
        if det["label"] in CRITICAL_LABELS and det["score"] >= threshold
    ]
    meta["critical_detections"] = critical_dets        # keep *only* critical

    is_critical = bool(critical_dets)                  # True = down‑link tile
    return is_critical, meta


if __name__ == "__main__":            
    results = run_inference(image_path=SINGLE_IMAGE_PATH,
                  folder_path=INFERENCE_FOLDER_PATH,
                  model_path=INFERENCE_MODEL_PATH,
                  threshold=CONFIDENCE_THRESHOLD,
                  single_infernce=SINGLE_INFERENCE
                  )
    print(results) # Metadata JSON output
    
    flag, filtered_metadata = filter_metadata(results, threshold=CONFIDENCE_THRESHOLD)
    print(f"Is critical: {flag}")
    print(f"Filtered metadata: {filtered_metadata}")
    

