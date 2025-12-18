Model Architecture and Design Philosophy
## Necati Deniz Baykus
## December 2025
## 1    Importance of the Model
The success of the emotion recognition model relies heavily on an architecture that
can extract discriminative features from a limited $64 \times 64$ pixel database while maintaining low  latency for real-time deployment.


## 2    Algorithm Survey and Model Selection
We evaluated three distinct architectural disciplines to identify the optimal balance between accuracy, parameter efficiency and  general capability:

Baseline CNN: A traditonal, shallow architecture utilizing a basic stack of convolutional, pooling, and fully connected layers. While computationally inexpensive and faster to train due to fewer layers and  relatively simple design, its performance is moderate, prone to overfitting and unable to successfully learn richer, more abstract representations. \cite{compCnnResnet}

Residual Networks (ResNet-18): This model introduces residual connections, which allow certain information to bypass certain layers, helping to avoid the vanishing gradient problem in deeper networks, presenting higher accuracy due to better feature learning. It is fairly complex due to added layers, skip connections and bottleneck desings which require their own respective implementations and careful tunings. It has high parameter efficiency due to the bottleneck structure reducing the number of parameters while preserving high performance. \cite{compCnnResnet2}

AlexNet: A fairly popular model,proposed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012, ALexNet  is  a relatively deep network structure consisting  of  eight learning layers, including five convolutional layers and three fully connected layers. Although has a higher parameter count compared to ResNet, it shows lower accuracy than Resnet and even basic CNN models which affected our final desicion.  \cite{compALexnetResnet}

Decision: We selected ResNet-18 (\ref{fig:diagram}) as our designated architecture due to its superior accuracy-to-parameter ratio and its native compatibility with Explainable AI (XAI) methods due to its more distributed attention over whole images and not relying on specific regions, which is useful for successful and versatile emotion recognition.


## 3   Proposed Architecture:  Modified ResNet-18
The standard ResNet-18 is designed for $224 \times 224$  samples. To accommodate our $64 \times 64$ input resolution, we propose several architectural modifications:

First, the initial convolution layer is changed. The standard $7 \times 7$ kernel  is replaced with a $3 \times 3$ kernel. This ensures that critical information is not discarded in the very first layer given the low-resolution input. \cite{resnetScratch}

The network then consists of four stages of residual blocks, each using Batch Normalization and ReLU activation, as in the original ResNet-18.

Instead of using a traditional Flatten layer followed by Fully Connected (FC) layers, we utilize Global Average Pooling (GAP). This reduces the total parameter count and makes the model robust against spatial translations of the face in the data.

Then, the final layer is a 6-node FC layer with Softmax activation, outputting the probability distribution across the target emotions: Happiness, Surprise, Sadness, Anger, Disgust, and Fear.


## 4 Hyperparameter Landscape 
The standard ResNet18 model utilizes the ReLU activation function,
ReduceROnPlateau as a learning rate scheduler, and Adam as an optimizer. However, in our research we found out that the ResNet18 model with Swish as the Activation
function, Adam as the Optimizer, and CosineAnnealingWarmRestarts as the Learning Rate Scheduler achieves the
highest testing accuracy, which will be the specifics of our ResNet-18 model. \cite{resnetparameter}

## 5 References

@article{compCnnResnet,
    author = {Milind Talele and Rajashree Jain  },
    title = {A Comparative Analysis of CNNs and
ResNet50 for Facial Emotion Recognition },
    journal = {Engineering, Technology & Applied Science Research},
    volume = 15,
    number = 2,
    year = 2024
}

@article{compAlexnetResnet,
    author = {Wenxuan Zhang  },
    title = {Comparison of AlexNet and ResNet Models for Remote Sensing Image Recognition },
    journal = {Transactions on Computer Science and Intelligent SystemsResearch},
    volume = 5,
    year = 2024
}

@article{compCnnResnet2,
    author = {Leonardo Reginato  },
    title = {A Practical Comparison Between CNN and ResNet Architectures: A Focus on Attention Mechanisms },
    journal = {Medium},
    year = 2024
}


@article{resnetScratch,
    author = {Rohit Modi },
    title = {ResNet — Understand and Implement from scratch},
    journal = {Medium},
    year = 2021
}


@article{resnetparameter,
    author = {Gaurav Kumar Pandeya
and Khushi Mittalb and Antra Bansalc
and Sumit Srivastavad },
    title = {Fire Detection with ResNet 18: Comparative Analysis Across Different Hyperparameters },
    journal = {ScienceDirect},
    year = 2025
}
