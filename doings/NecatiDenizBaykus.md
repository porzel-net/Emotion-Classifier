## Necati Deniz Baykuş: Contributions to the Emotion-Classifier Project

### Core Model Design
- Implemented the main emotion recognition architecture based on ResNet-18.
- Adapted the backbone to a lightweight configurable structure suitable for VM-scale training.
- Implemented the multi-branch feature architecture combining:
  - CNN backbone representation
  - edge-based feature branch (Sobel)
  - fused feature classification head
- Implemented configurable model scaling via width multiplier and dropout control.

### Model Engineering and Optimization
- Built the full forward pipeline and feature fusion logic.
- Tuned architectural parameters for stability and performance under resource constraints.
- Ensured the model runs end-to-end on real datasets within the project pipeline.

### System-Level Model Integration
- Integrated the model into the project training framework.
- Verified compatibility with dataset loaders and preprocessing pipeline.
- Validated model outputs and ensured successful execution in the VM environment.
- Ensured metadata and landmark data were correctly consumed by the model.
- Monitored training metrics (loss, accuracy, F1) across epochs.
- Served as the primary executor of computational experiments for the project.
