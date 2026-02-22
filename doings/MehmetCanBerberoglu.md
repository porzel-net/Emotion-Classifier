## Mehmet Can Berberoğlu — Contributions to the Emotion-Classifier Project

### Training Pipeline and Execution
- Implemented the core training loop structure including forward, backward, and optimization steps.
- Designed and maintained the epoch-based training workflow and scheduler integration.
- Implemented dataset loading and reloading logic for stable training across experiments.
- Co-managed VM-based training runs and experiment execution.

### Data Handling and Loaders
- Implemented dataset preparation structure using ImageFolder-based pipelines.
- Built training and evaluation DataLoader configurations with batching, shuffling, and memory optimization.
- Implemented hold-out dataset splitting utilities (train/test separation).

### Evaluation and Metrics
- Implemented model evaluation utilities including:
  - compute_accuracy for batch-wise evaluation
  - prediction extraction pipeline for post-training analysis
- Merged the custom metric functions into the main code:
  - accuracy
  - macro precision
  - macro recall
  - macro F1 score
- Built full evaluation pipeline producing performance metrics after training.

### Training Utilities and Experiment Support
- Implemented prediction extraction utilities for metric computation.
- Contributed to experiment reproducibility via deterministic setup and evaluation flow.
- Actively participated in VM training processes and experimental runs.
