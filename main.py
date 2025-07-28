from scripts.inference import run_inference, filter_metadata
from config import (
    CONFIDENCE_THRESHOLD,
    SINGLE_INFERENCE,
    INFERENCE_MODEL_PATH,
    SINGLE_IMAGE_PATH,
    INFERENCE_FOLDER_PATH,
)

def main():
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


if __name__ == "__main__": 
    main()          
