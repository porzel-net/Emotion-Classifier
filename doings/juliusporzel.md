# Julius Porzel Doings in the Emotion-Classifier Project

## Project Setup and Data Pipeline
- Maintained the overall project structure (including several major refactors and script updates to match model architecture changes).
- Created and standardized dataset setup via `data/download_datasets.sh`.
- Implemented FER2013 preprocessing (`prepare_data/prepare_fer2013.py`) and AffectNet preprocessing (`prepare_data/prepare_affectnet.py`):
  - resize, landmark detection, metadata export (`metadata.csv`), class mapping, split processing, landmark metadata.
  - robust data preparation (handling corrupted images and missing landmarks).
- Enabled multi-dataset domain combination, including a separate validation domain via `--val-data-root` and `--val-data-split`.

## K-Means Baseline and Clustering Experiments
- Implemented K-Means baseline (`train/train_kmeans_baseline.py`).
- Tested advanced clustering approaches:
  - K-Means++ with multi-restarts,
  - spherical K-Means,
  - L2 normalization,
  - PCA + whitening,
  - landmark fusion in feature space,
  - cluster-to-class mapping (majority vote / optimal 1:1 assignment).

## Independent Preliminary Experiments
- Trained a custom face cropper (`train/train_cropping.py`):
  - multi-task setup (face confidence + bounding box),
  - backbone comparison (custom lightweight separable-conv residual backbone vs. MobileNetV3Small),
  - optional backbone freezing to test transfer behavior vs. full end-to-end tuning,
  - boundary-aware loss and boundary undersampling.
  - weighted multi-objective optimization (`confidence_loss_weight` + `bbox_loss_weight`) to balance detection confidence vs. localization quality.
  - tested translation-aware bbox jitter with clipping for stable in-frame box generation.
  - tested zoom-aware bbox jitter to improve robustness to face-size variance.
  - tested many variants of architectures.
- Trained a custom landmark detector (`train/train_landmarks_detector.py`):
  - compact regressor,
  - direct coordinate-regression design (landmark vector output) as a lightweight alternative to full heatmap decoders,
  - configurable augmentation rounds to synthetically increase geometric diversity,
  - tested all kinds of different architectures

## Modeling and Training
- Refactored emotion-classifier training and added own model approaches (`train/train_emotion_classifier.py`).
- Integrated landmark input as an optional additional channel.
- Implemented optional landmark GNN branch.
- Implemented multi-view landmark attention / heatmap branch.
- Added an optional Sobel side branch.
- Implemented multi-branch fusion (CNN + landmark + heatmap + GNN).

## Methodological Approaches and Ablations
- Tested filter ablations: `none`, `sobel`, `roberts`, `laplacian`.
- Tested sampling ablations: `original`, `undersample`, `loss-weight`.
- Tested augmentations:
  - rotation, scale-crop, translation, perspective+shear, random erasing, Gaussian noise.
- Implemented landmark-synchronized transformation logic for augmentation.

## Inference, Visualization, Explainable AI
- Implemented live webcam/video inference (`visualize/predict_emotion.py`).
- Integrated explainable AI modes (`visualize/predict_emotion.py`).
- Built visualization for cropper predictions (`visualize/display_cropping_predictions.py`).
- Built visualization for landmark predictions (`visualize/display_landmarks_prediction.py`).
- Built preview tool for emotion-classifier predictions (`visualize/show_emotion_classifier_prediction.py`).
- Wrote the confusion-matrix script (`scripts/evaluate_confusion_matrix.py`).
- Implemented folder-to-CSV scoring (`scripts/score_folder.py`).

## Reports and Documentation
- Researched the data strategy for the Preliminary Report.
- Wrote the Preliminary Report (except metrics section).
- Built and extended the Final Report (as team with Memo and Necati):
  - Method & Approach, Experiments, Results, Discussion/Conclusion/Outlook.
- Refactored the project comprehensively for submission (including `README.md`, etc.).
