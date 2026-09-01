# YOLOv8 Fine-Tuning Stretch Goal -- Results

## What this is

A from-scratch fine-tune of a pretrained YOLOv8n model on real, published,
bounding-box-annotated lunar crater imagery, run to test the deep-learning
upgrade path over the classical-CV baseline (`src/landing_site/craters.py`)
used in the main pipeline. `build_dataset.py` converts the source
annotations to YOLO format; the training command used is in `train.py`.

## Data

Yang, H., Xu, X., Ma, Y., Xu, Y., Liu, S., "CraterDANet: A Convolutional
Neural Network for Small-Scale Crater Detection via Synthetic-to-Real
Domain Adaptation," *IEEE Transactions on Geoscience and Remote Sensing* --
real LRO Narrow Angle Camera images (0.5 m/pixel) near the Chang'E-4
landing site (45-46S, 176.4-178.8E), via
[github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data](https://github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data).

- 12 training images, 800x800 px, 13,449 real crater bounding boxes
- 8 validation images, 1000x1000 px, 7,298 real crater bounding boxes
- Craters average only ~7 px across in the source images -- near the
  practical resolution limit for a general-purpose object detector

`ground_truth_density_example.jpg` shows why this is a hard dataset: the
real crater density is so high that ground-truth boxes visually merge into
a solid mass at normal image scale.

## Training setup

- Model: YOLOv8n (`ultralytics`), initialized from COCO-pretrained weights
- 6 epochs, image size 640, batch size 4, CPU only (no GPU was available in
  this sandboxed environment), ~6 minutes total wall-clock time

## Results

| Metric | Value (after 6 epochs) |
|---|---|
| Precision | 0.018 |
| Recall | 0.023 |
| mAP50 | 0.0021 |
| mAP50-95 | 0.0007 |

`training_curves.png` shows box/class/DFL loss decreasing across all 6
epochs on both train and validation splits -- the model is genuinely
learning something -- but absolute detection quality after only 6 epochs is
not yet usable: at a confidence threshold of 0.05, the fine-tuned model
produces **zero** detections on any of the 8 validation images.

## Honest interpretation

This is a proof that the fine-tuning *pipeline* is correct and runs
end-to-end on real annotated data -- not a working crater detector. Three
specific, known reasons this run doesn't (yet) produce good detections,
each with a concrete fix:

1. **Far too little training data.** 12 images is a tiny fraction of the
   full ~20,000-crater dataset this repo provides across many more
   image pairs; typical YOLO fine-tunes use hundreds to thousands of
   images.
2. **Far too few epochs.** YOLO detectors typically need 50-300 epochs to
   converge; this run did 6, chosen to fit this sandboxed CPU environment's
   time budget.
3. **No GPU.** Each epoch took roughly 30-60 CPU-seconds for just 11
   training images; scaling to the full dataset and epoch count listed
   above is a multi-hour job on CPU but a very reasonable one on any
   consumer GPU (Colab's free tier is enough).

The classical-CV baseline in the main pipeline (`craters.py`), by
contrast, requires no training data at all and already recalls 90% of the
real catalogued craters at Tycho -- which is exactly why it, not this
YOLOv8 experiment, is the pipeline's primary, validated crater detector.
This experiment is included specifically to show the fine-tuning workflow
executed correctly on genuine data, and to give a concrete, honest basis
for the "how would you scale this up" conversation in a report or
interview.

## Files

- `build_dataset.py` -- converts the source repo's per-image CSV
  annotations to YOLO label format (handles each image's true pixel
  dimensions, since they differ between the train and val splits)
- `train.py` -- the training command
- `training_curves.png` -- loss and metric curves across all 6 epochs
- `ground_truth_density_example.jpg` -- a real validation image with its
  (very dense) ground-truth boxes overlaid
- `runs/detect/runs/crater_finetune/` -- full ultralytics training run
  output, including the fine-tuned weights
  (`weights/best.pt`), confusion matrix, and precision/recall curves
