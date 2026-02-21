# 1. Training Strategy

This section describes how the facial emotion classification model is planned to be trained. 
The overall training setup is mainly influenced by two factors: the fixed low input resolution of 64×64 pixels and the imbalance between the six emotion classes. 
Both aspects make the training process more sensitive and require some additional care compared to standard image classification tasks.

## 1.1 Training Pipeline

After all preprocessing steps are completed, the processed face images will be used as input for the modified ResNet-18 model. 
For each image, the network outputs a set of probabilities, one for each emotion class. 
These outputs are then compared with the corresponding labels to calculate the training loss.

Training will be performed over several epochs, since facial expressions cannot be learned reliably from a single pass over the data. 
After each batch, the model parameters will be updated using backpropagation. 
Alongside the training data, a validation split will be used to get a better idea of how well the model generalises to unseen samples. 
This is especially important in facial emotion recognition, where models can easily start learning dataset-specific patterns instead of expression-related features which might lead to overfitting.

## 1.2 Dealing with Class Imbalance

One of the main difficulties in this task is the uneven distribution of emotion classes. 
In some facial emotion datasets, some emotions appear much more often than others.

To reduce this bias, a weighted cross-entropy loss function will be used during training. 
Emotions that occur less frequently will be assigned higher weights so that their classification errors have a stronger impact on the optimisation process. 
This approach encourages the model to pay attention to all classes rather than mainly optimising for the most common ones. 
If this turns out not to be sufficient, additional balancing techniques such as modified sampling strategies may be considered.

# 2. Optimisation Strategy

This section outlines how the model parameters will be optimised during training.

## 2.1 Optimiser Selection

The Adam optimiser will be used as the main optimisation method. 
Adam is a common choice in deep learning because it adjusts parameter updates automatically, which can be helpful when training on data with varying image quality, lighting conditions, and expression intensity.
As an alternative, stochastic gradient descent with momentum may be explored if more direct control over the optimisation behaviour is needed.

## 2.2 Activation Functions and Regularisation

ReLU will be used as the activation function throughout the network. 
It is widely used in convolutional neural networks because it is simple, efficient, and works well in deeper architectures such as ResNet-18. 
In practice, ReLU also helps reduce problems related to vanishing gradients.

To limit overfitting, several regularisation methods will be applied. 
Dropout will be used in the later layers to prevent the model from relying too strongly on individual neurons. 
In addition, L2 weight decay will be applied to keep the model parameters from growing too large. 
These techniques are intended to improve generalisation, especially given the relatively small and imbalanced nature of facial emotion datasets.

## 2.3 Summary

In summary, the planned training and optimisation approach aims for stable and balanced learning rather than aggressive fine-tuning. 
By combining imbalance-aware loss functions, adaptive optimisation, ReLU activations, and regularisation techniques, the model should be able to learn relevant emotion-related facial features while avoiding common training issues.
