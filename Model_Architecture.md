# Model Architecture and Design Philosophy
**Author:** Necati Deniz Baykuş  
**Date:** December 2025

## 1. Introduction
The success of the emotion recognition model relies heavily on an architecture that can extract discriminative features from a limited $64 \times 64$ pixel budget while maintaining low inference latency for real-time deployment.

## 2. Algorithm Survey and Model Selection
We evaluated three distinct architectural paradigms to identify the optimal balance between depth, parameter efficiency, and generalization capability:

* **Baseline CNN (LeNet-style):** A shallow architecture utilizing 2-3 convolutional layers. While computationally inexpensive, it lacks the representational power to capture the subtle micro-expressions of the human face required for 6-class classification.
* **VGG-16 / AlexNet:** These models use large filter banks and deep stacks of convolutions. However, their high parameter count (up to 138M) poses a significant risk of overfitting on specialized datasets like FER-2013.
* **Residual Networks (ResNet-18):** This model introduces **skip connections** (shortcuts) that bypass one or more layers. Instead of learning an entirely new mapping, the network learns the residual (the difference) between the input and output. This "identity mapping" ensures that critical facial features are preserved across deep layers.

**Decision:** We selected **ResNet-18** as our core architecture due to its superior accuracy-to-parameter ratio and its native compatibility with Explainable AI (XAI) methods.



---

## 3. Proposed Architecture: Modified ResNet-18
The standard ResNet-18 is designed for $224 \times 224$ ImageNet samples. To accommodate our $64 \times 64$ input resolution, we propose several architectural modifications:

1.  **Initial Convolution Layer:** The standard $7 \times 7$ kernel (stride 2) is replaced with a $3 \times 3$ kernel (stride 1) to preserve spatial information in low-resolution input.
2.  **Feature Extraction Blocks:** Four stages of residual blocks using Batch Normalization and ReLU activation.
3.  **Global Average Pooling (GAP):** Replaces traditional Flatten layers to reduce parameter count and increase robustness against spatial translations.
4.  **Softmax Classifier:** A 6-node FC layer outputting probabilities for: *Happiness, Surprise, Sadness, Anger, Disgust, and Fear.*

---

## 4. Hyperparameter Landscape and Search Space
To optimize the training process, we have defined the following parameters:

* **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$) with a fallback to SGD with Momentum.
* **Learning Rate Strategy:** Base rate of $\eta = 1e-3$ with a *ReduceLROnPlateau* scheduler.
* **Regularization:** Dropout rate of $0.3$ and $L_2$ Weight Decay of $1e-4$.
* **Loss Function:** **Weighted Cross-Entropy Loss** to compensate for class imbalance (e.g., the minority "Disgust" class).

---

## 5. Design for Interpretability (XAI)
A primary requirement is to provide visual explanations using **Grad-CAM** (Gradient-weighted Class Activation Mapping). By maintaining spatial integrity until the GAP layer, we back-propagate gradients to generate **Saliency Maps**. These maps highlight which facial regions (e.g., eyes for Surprise, mouth for Happiness) the model prioritized.


## 6. References

* **[1] Lopes, A. T., et al. (2016).** "Facial expression recognition with Convolutional Neural Networks: Coping with few data and the training sample order." *Elsevier*. (Discusses the limitations of shallow networks and data constraints).
* **[2] Li, B., & Lima, D. (2021).** "Facial expression recognition via ResNet-50." *Keiapublishing*. (Basis for selecting Residual Networks for emotion classification).
* **[3] Jafar, A., & Lee, M. (2021).** "High-speed hyperparameter optimization for deep ResNet models in image recognition." *Springer Nature*. (Supports our selection of Adam optimizer and learning rate strategies).


---
