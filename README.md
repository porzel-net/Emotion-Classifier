# Emotion Classifier (SEP Project)

## Overview
- Build an emotion recognition pipeline tailored to the given deliverables.
- Core goal: classify six facial emotions (happiness, surprise, sadness, anger, disgust, fear) from 64×64 images and support interpretability.

## Current Task
1. **Classification Model** – Train a PyTorch model from scratch; evaluate accuracy on held-out test data.
2. **Explainable AI** – Provide visual explanations (saliency, CAM/GradCAM, occlusion, etc.) highlighting regions the model uses.
3. **Demo** – Process an input video (starting from provided footage, later a webcam stream if feasible), overlay emotion predictions and saliency maps, and save the annotated result.

## Deliverables
- Preliminary report (2 pages, double-column using provided LaTeX template) covering approach, architectures, literature review, and initial findings.
- Presentation (15 min including demo) with each member covering a section; questions from instructors and peers expected.
- Final report (8 pages double-column) structured like a scientific paper (Introduction, Related Work, Method, Experiments) plus a contributions appendix.
- Codebase in PyTorch with:
  - A README (this file) explaining setup and task scope.
  - A script that consumes an image-folder path and outputs per-image emotion scores to CSV.
  - A demo script that ingests a video file, classifies frames, visualizes important regions, and saves the annotated video.

## Setup
- Install dependencies from `requirements.txt` (ideally inside a virtual environment such as `conda` or `venv`).
- Prepare the dataset externally (do not upload to repo); document download/preprocessing steps separately (data size limit applies).

## Next Steps
- Finalize model architecture and training pipeline.
- Implement explainability visualizations and demo workflow.
- Align reports/presentation materials with the implemented system and recorded results.
