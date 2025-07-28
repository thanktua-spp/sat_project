from datetime import datetime
from pathlib import Path

import modal

from config import PROJECT_NAME

volume = modal.Volume.from_name(PROJECT_NAME, create_if_missing=True)
volume_path = (
    Path("/xview_rfdetr")
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(  # install system libraries for graphics handling
        ["libgl1-mesa-glx", "libglib2.0-0"]
    )
    .pip_install(  # install python libraries for computer vision            "datasets>=4.0.0",
            "opencv-python-headless>=4.12.0.88",
            "pandas>=2.3.1",
            "rfdetr>=1.2.1",
            "rich>=14.0.0",
            "sahi>=0.11.31",
            "torch>=2.7.1",
            "torchaudio>=2.7.1",
            "torchvision>=0.22.1"
            )
    .pip_install(  # add an optional extra that renders images in the terminal
        "term-image==0.7.1"
    )
)


app = modal.App(PROJECT_NAME, image=image, volumes={volume_path: volume})


MINUTES = 240

TRAIN_GPU_COUNT = 1
TRAIN_GPU = f"A100:{TRAIN_GPU_COUNT}"
TRAIN_CPU_COUNT = 4


@app.function(
    gpu=TRAIN_GPU,
    cpu=TRAIN_CPU_COUNT,
    timeout=60 * MINUTES,
)
def train(
    model_id: str,
    num_epochs: int,
):
    from rfdetr import RFDETRNano

    volume.reload()  # make sure volume is synced
    
    model = RFDETRNano()
    model.train(
        # dataset config
        dataset_dir="/xview_rfdetr" + str(volume_path),
        epochs=num_epochs,
        batch_size=16,
        grad_accum_steps=1,
        lr=1e-4,
        early_stopping=True,
        gradient_checkpointing=True,
        output_dir="/xview_rfdetr/model10_checkpoints/" + model_id,
        #resume=str("/xview_rfdetr/model10_checkpoints/runsxview_rfdetr20250727_054755/checkpoint.pth"),
    )

@app.local_entrypoint()
def main():
    #train.map(range(4))
    train.remote(
        model_id=f"xview_rfdetr{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        num_epochs=30,
    )
    print("Training started. Check the Modal dashboard for progress.")