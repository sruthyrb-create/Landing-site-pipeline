"""
Fine-tunes a pretrained YOLOv8n model on the crater dataset built by
build_dataset.py. See RESULTS.md for the actual metrics obtained from this
exact command in a CPU-only, time-constrained environment, and for what
scaling it up (more data, more epochs, a GPU) would need to change.

Usage:
    pip install ultralytics
    python3 build_dataset.py /path/to/Lunar_Crater_Detection_Data
    python3 train.py
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # COCO-pretrained weights, downloaded automatically
    model.train(
        data="dataset/data.yaml",
        epochs=6,        # RESULTS.md used 6 to fit a CPU-only time budget; use 50-300+ with a GPU
        imgsz=640,
        batch=4,
        device="cpu",    # set to 0 (or omit) to use a GPU if available
        workers=0,
        project="runs",
        name="crater_finetune",
        exist_ok=True,
        patience=0,
    )
