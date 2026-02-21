# Emotion Classifier

The repository contains a full facial-emotion pipeline built around:
- dataset preparation,
- model inference for images/videos/webcam,
- explainable AI overlays,
- training scripts for a custom face cropper and landmark detector,
- training script using K-Means for emotion classifying (first try of what is possible with pure K-Means).
- training script for cnn emotion classifier

The main emotion classes are:
`happiness`, `surprise`, `sadness`, `anger`, `disgust`, `fear`.

## Project Structure

- `train/`: model training and training/evaluation experiments
- `visualize/`: prediction viewers and interactive visualizations
- `scripts/`: dataset prep and utility scripts
- `helpers/`: shared model/data helper modules

## 1. Data Setup

Run from the repository root.

### One command for all datasets

```bash
bash data/download_datasets.sh
```

What this does:
- downloads missing archives only (no double-download),
- unzips automatically,
- ensures stable folder names used by defaults in the scripts.

Prepared folders in `data/`:
- `fer2013`
- `cropped-face-keypoint-dataset-68-landmarks`
- `soloface-detection-dataset`
- `affectnet-yolo-format`

### Optional single-dataset calls

```bash
bash data/download_datasets.sh --dataset fer2013
bash data/download_datasets.sh --dataset landmarks68
bash data/download_datasets.sh --dataset soloface
bash data/download_datasets.sh --dataset affectnet
```

Im `data/`-Ordner gibt es nur dieses eine Setup-Skript:
- `data/download_datasets.sh`

## 2. Model Weights

Download `Neconet_Weights3.pth` from your TODO URL and place it in:

- `models/Neconet_Weights3.pth`

(Or pass a custom weights path via CLI where supported.)

## 3. Build the Modified FER Dataset

Script: `prepare_data/prepare-fer2013.py`

Purpose:
- reads FER data from `train/` and `test/` class folders,
- detects facial landmarks,
- writes processed images plus `metadata.csv` with normalized landmarks.

Example:

```bash
python prepare_data/prepare-fer2013.py --device cpu
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--source` | No | `data/fer2013` | Input FER root (must contain `train/` and `test/`). |
| `--target` | No | `data/emotion-classifier-dataset` | Output root for processed dataset and metadata. |
| `--resize` | No | `48` | Output image size (`resize x resize`). |
| `--device` | No | `cpu` | Device used by `face_alignment` (`cpu`, `mps`, `cuda`). |

## 4. Folder Scoring to CSV

Script: `scripts/score_folder.py`

Purpose:
- classifies all images in one folder,
- writes one CSV row per image,
- outputs class probabilities.

Example:

```bash
python -m scripts.score_folder ./data/emotion-classifier-dataset/test/happy/ \
  --weights-dir models/emotion-classifier-best.pth \
  --device mps
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `images` (positional) | Yes | - | Folder containing images to score. |
| `--output`, `-o` | No | `predictions.csv` | Output CSV file path (overwrites if exists). |
| `--weights`, `--weights-dir` | No | `models/emotion-classifier-best.pth` | Checkpoint file path. |
| `--device` | No | auto | Torch device (`cpu`, `mps`, `cuda`). |
| `--filter` | No | `sobel` | Input preprocessing (`sobel`, `roberts`, `laplacian`, `none`). |
| `--width-multiplier` | No | `1.0` | Width multiplier used in the trained model architecture. |
| `--use-landmarks` | No | `False` | Enables landmark input branch. |
| `--use-gnn` | No | `False` | Enables GNN processing over landmarks. |
| `--gnn-hidden-dim` | No | `64` | GNN hidden dimension (must match training). |
| `--gnn-steps` | No | `2` | Number of GNN message-passing steps. |
| `--metadata-file` | No | `data/emotion-classifier-dataset/metadata.csv` | Landmark CSV used when `--use-landmarks` is enabled. |
| `--metadata-root` | No | `.` | Root path for resolving metadata image paths. |

## 5. Live Webcam/Video Prediction with explainable AI

Script: `visualize/predict_emotion.py`

Purpose:
- runs emotion inference from webcam or video,
- draws face box, landmarks, probabilities,
- optionally overlays explainability heatmaps (`saliency`, `activations`, `occlusion`, `cam`, `gradcam`).

Example (webcam):

```bash
python -m visualize.predict_emotion --webcam --weights models/Neconet_Weights3.pth --device mps --explain gradcam
```

Example (video input + output):

```bash
python -m visualize.predict_emotion --video video.mov --output video-out.mov --explain saliency
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--weights`, `--weights-dir` | No | `models/Neconet_Weights3.pth` | Model checkpoint path. |
| `--device` | No | auto | Torch device (`cpu`, `mps`, `cuda`). |
| `--webcam` | Conditionally | `False` | Use webcam input. Required if `--video` is not set. |
| `--video` | Conditionally | - | Input video path. Required if `--webcam` is not set. |
| `--output` | No | auto | Output video path for video mode (`<input>_annotated.mp4` if omitted). |
| `--scale` | No | `0.4` | Downscale factor for landmark detection stage. |
| `--skip-frames` | No | `2` | Run landmark detection every Nth frame. |
| `--show` | No | `False` | Show processed frames while running (video mode). |
| `--explain` | No | - | Explanation mode: `saliency`, `activations`, `occlusion`, `cam`, `gradcam`. |
| `--occlusion-patch` | No | `8` | Patch size for occlusion explanation. |
| `--occlusion-stride` | No | `4` | Patch stride for occlusion explanation. |

## 6. Train a Custom Face Cropper

Script: `train/train_cropping.py`

Purpose:
- trains a model that predicts face confidence and bounding box,
- supports lite or MobileNetV3 backbone,
- supports augmentation and boundary-aware losses,
- and more...

Example:

```bash
python -m train.train_cropping \
  --data-dir data/soloface-detection-dataset \
  --train-split train \
  --val-split val \
  --output models/solo_cropper_v2.keras \
  --image-size 64 \
  --epochs 40 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --augment \
  --augment-translation 10 \
  --augment-zoom 0.1 \
  --backbone mobilenet_v3_small
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--data-dir` | No | `data/soloface-detection-dataset` | SoloFace dataset root. |
| `--train-split` | No | `train` | Training split subfolder. |
| `--val-split` | No | `val` | Validation split subfolder. |
| `--output` | No | `models/solo_cropper.keras` | Best-checkpoint output path. |
| `--image-size` | No | `64` | Input size (square). |
| `--epochs` | No | `30` | Number of training epochs. |
| `--batch-size` | No | `64` | Batch size. |
| `--learning-rate` | No | `1e-3` | Initial optimizer LR. |
| `--max-train-samples` | No | unlimited | Cap train samples for quick runs. |
| `--max-val-samples` | No | unlimited | Cap validation samples. |
| `--seed` | No | `42` | Random seed. |
| `--augment` | No | `False` | Enables augmentation pipeline. |
| `--augment-translation` | No | `10.0` | Translation magnitude in percent (converted internally). |
| `--augment-zoom` | No | `0.1` | Zoom factor for augmentation. |
| `--augment-flip` / `--no-augment-flip` | No | flip enabled | Toggle horizontal flip when augmentation is active. |
| `--augment-color-jitter` / `--no-augment-color-jitter` | No | jitter enabled | Toggle brightness/contrast jitter when augmentation is active. |
| `--backbone` | No | `lite` | Feature extractor: `lite` or `mobilenet_v3_small`. |
| `--freeze-backbone` | No | `False` | Freezes backbone weights. |
| `--dropout` | No | `0.2` | Dropout rate before heads. |
| `--confidence-loss-weight` | No | `0.8` | Weight for confidence loss. |
| `--bbox-loss-weight` | No | `1.2` | Weight for bbox regression loss. |
| `--bbox-loss-delta` | No | `0.1` | Huber delta for bbox loss. |
| `--boundary-penalty` | No | `0.0` | Extra penalty weight near/outside boundaries. |
| `--boundary-penalty-threshold` | No | `0.0` | Boundary distance threshold for penalty scaling. |
| `--undersample-boundary-threshold` | No | `0.0` | Distance threshold to mark near-boundary samples. |
| `--undersample-boundary-rate` | No | `0.0` | Fraction of near-boundary samples to drop. |
| `--disable-reduce-lr` | No | `False` | Disables `ReduceLROnPlateau` callback. |

## 7. K-Means Baseline Evaluation

Script: `train/train_kmeans_baseline.py`

Purpose:
- runs clustering-only baseline on flattened image features,
- supports spherical K-Means,
- supports optional landmark feature fusion,
- reports clustering-based classification metrics.

Training checkpoints from `python -m train.train_emotion_classifier` are written to `models/`, with the current best model at `models/emotion-classifier-best.pth`.

Example:

```bash
python -m train.train_kmeans_baseline \
  --filter sobel \
  --k 6 \
  --normalize-features \
  --pca-components 128 \
  --use-landmark-fusion \
  --landmark-weight 1.2 \
  --spherical-kmeans \
  --n-init 20
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--data-root` | No | `data/emotion-classifier-dataset` | Dataset root containing `train/` and `test/`. |
| `--filter` | No | `sobel` | Preprocessing filter (`sobel`, `roberts`, `laplacian`, `none`). |
| `--k` | No | number of classes (6) | Number of clusters. |
| `--max-iter` | No | `100` | Max iterations per run. |
| `--tol` | No | `1e-3` | Convergence tolerance. |
| `--batch-size` | No | `256` | Batch size for feature extraction/assignment. |
| `--seed` | No | `1` | Random seed. |
| `--n-init` | No | `10` | Number of K-Means restarts. |
| `--spherical-kmeans` | No | `False` | Use cosine similarity variant. |
| `--max-train-samples` | No | unlimited | Optional train sample cap. |
| `--max-test-samples` | No | unlimited | Optional test sample cap. |
| `--normalize-features` | No | `False` | Apply L2 normalization before clustering. |
| `--no-balance-train` | No | `False` | Disable class-balanced train subsampling. |
| `--balance-max-per-class` | No | unlimited | Cap per class when balancing. |
| `--no-pca-whitening` | No | `False` | Disable PCA+whitening projection. |
| `--pca-components` | No | `128` | PCA output dimension (if PCA enabled). |
| `--no-optimal-assignment` | No | `False` | Disable optimal cluster-to-class mapping. |
| `--use-landmark-fusion` | No | `False` | Concatenate landmark vectors from metadata. |
| `--metadata-file` | No | `data/emotion-classifier-dataset/metadata.csv` | Landmark metadata CSV. |
| `--landmark-weight` | No | `1.0` | Scaling factor for landmark feature block. |
| `--report-json` | No | disabled | Optional path for JSON report output. |

## 8. Train a Custom Landmark Detector

Script: `train/train_landmarks_detector.py`

Purpose:
- trains a lightweight landmark regression model,
- supports geometric/noise augmentation,
- supports optional feature standardization and LR scheduling.

Example:

```bash
python -m train.train_landmarks_detector \
  --data-dir data/cropped-face-keypoint-dataset-68-landmarks \
  --epochs 60 \
  --batch-size 64 \
  --augmentation-rounds 2 \
  --max-rotation 15 \
  --min-scale 0.8 \
  --max-translation 0.12 \
  --noise-prob 0.4 \
  --noise-scale 0.03 \
  --standardize \
  --reduce-lr plateau \
  --reduce-lr-factor 0.5 \
  --reduce-lr-patience 4 \
  --reduce-lr-min-lr 1e-6
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--data-dir` | No | `data/cropped-face-keypoint-dataset-68-landmarks` | Dataset root. |
| `--csv` | No | `training.csv` | Landmark CSV (relative to `--data-dir`). |
| `--images` | No | `training` | Image folder (relative to `--data-dir`). |
| `--output` | No | `models/landmarks_detector.keras` | Best model output path. |
| `--epochs` | No | `40` | Training epochs. |
| `--batch-size` | No | `32` | Batch size. |
| `--learning-rate` | No | `1e-3` | Initial optimizer LR. |
| `--val-split` | No | `0.15` | Validation fraction. |
| `--seed` | No | `42` | Random seed. |
| `--max-samples` | No | unlimited | Optional dataset cap for experiments. |
| `--augmentation-rounds` | No | `1` | Number of augmented copies per sample. |
| `--max-rotation` | No | `10.0` | Rotation range in degrees. |
| `--min-scale` | No | `0.85` | Minimum zoom scale in augmentation. |
| `--max-translation` | No | `0.1` | Translation range relative to image size. |
| `--noise-prob` | No | `0.5` | Probability of adding Gaussian noise per augmented sample. |
| `--noise-scale` | No | `0.02` | Noise standard deviation. |
| `--standardize` / `--no-standardize` | No | standardize enabled | Toggle pixel standardization. |
| `--reduce-lr` | No | `plateau` | LR scheduler mode: `plateau` or `none`. |
| `--reduce-lr-factor` | No | `0.5` | Plateau LR reduction factor. |
| `--reduce-lr-patience` | No | `3` | Plateau patience (epochs). |
| `--reduce-lr-min-lr` | No | `1e-6` | Minimum LR under plateau scheduler. |

## 9. Notes

- `visualize/predict_emotion.py` requires either `--webcam` or `--video`.
- `scripts/score_folder.py` expects a folder (not recursive).
- For landmark-aware inference/evaluation, ensure `metadata.csv` exists from dataset preparation.
