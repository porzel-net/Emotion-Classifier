# Data Strategy Facial Emotion Classification

This report details our advanced methodology to train an AI model by a fixed $64 \times 64$ input resolution and a six-emotion classification goal.

---

## 1. Foundational Data Constraints and Challenges

**Main challenges**: low image resolution and the resulting class imbalance.

### 1.1. Resolution Bottleneck

* The fixed $64 \times 64$ resolution severely limits the detail available to capture subtle micro-expressions, which are essential for distinguishing emotions like **fear from surprise** [[Data Distribution of the New Balanced FER2013 Dataset]](https://www.researchgate.net/figure/Data-Distribution-of-the-New-Balanced-FER2013-Dataset_fig4_352846372).
* At this low resolution, the model cannot effectively learn to ignore (become invariant to) changes in face pose, angle, or scale.
* To compensate, **geometric variations must be physically removed** from the input data before the images are fed to the network. This ensures the network's processing power focuses only on expressive dynamics, not pose corrections [[RetinaFace: Single-shot Multi-level Face Localisation in the Wild]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.pdf).
* the primary dataset, FER-2013, uses $48 \times 48$ images, requiring a mandatory upscaling step to meet the $64 \times 64$ requirement, which must be executed using advanced techniques to avoid introducing blurring artifacts.

### 1.2. The 6-Class Taxonomy and Class Imbalance

* Standard FER datasets like FER-2013 and AffectNet use seven classes (Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral). The exclusion of the common "Neutral" class radically changes the dataset distribution.
* By removing the large "Neutral" category, the inherent class imbalance among the remaining six emotional categories (**Happy is often abundant, while Disgust or Fear are rare**) becomes significantly magnified [[Introducing a novel dataset for facial emotion recognition and demonstrating significant enhancements in deep learning performance through pre-processing techniques]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11620061/).
* Training the model using conventional random sampling and standard cross-entropy loss will result in the model favoring the majority classes, leading to **poor generalization** and low accuracy on the critical minority emotions [[Dynamically Weighted Balanced Loss: Class Imbalanced Learning Dynamically Weighted Balanced Loss: Class Imbalanced Learning and Confidence Calibration of Deep Neural Networks]](https://digitalcommons.usf.edu/cgi/viewcontent.cgi?article=1032&context=mth_facpub).

---

## 2. Multi-Source Data Acquisition and Roles

To achieve competitive generalization (robustness on unseen test data), the model could't be trained on diverse data sources, not just one.

| Dataset                                                                           | Key Characteristics                                                                                                                      |
|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|
| **[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)**                  | $48 \times 48$ grayscale images ($\sim40,000$), structured environment.                                                                  |
| **[AffectNet-8](https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format)** | Massive size of 96x96 images ($\sim450,000$), "in-the-wild" diversity (varied lighting, pose, background), **includes facial landmarks** |
| **[RAF-DB](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset/data)**       | $\sim15,000$ images, real-world complexity.                                                                                              |

*Key problems when combining different datasets: different sizes of datasets and the basic image differences, e.g. color gradations etc.*

---

## 3. Advanced Preprocessing Pipeline for $64 \times 64$ Success

The images could pass through a mandatory, three-stage preprocessing pipeline to ensure the limited pixel budget is used efficiently.

### 3.1. Face Detection

* Using **RetinaFace** for superior precision for especially extracting the face out of a bigger picture (will be important for the last task, the live emotion classifier) [[RetinaFace]](https://github.com/serengil/retinaface).
* RetinaFace locates the face bounding box and extracts five crucial facial landmarks: the inner corners of both eyes, the tip of the nose, and the outer corners of the mouth [[RetinaFace]](https://github.com/serengil/retinaface).
* **If AffectNet-8 is used, it might make sense to train a new own facial landmarks detector in order to obtain consistent data for later forwards (facial landmarks are coming with AffectNet-8 dataset).**

### 3.2. Geometric Normalization (Face Stabilization)

* **Algorithm:** A **Similarity Transform** (a combination of translation, rotation, and scaling) is computed using precise detected landmarks.
* **Process:** This transform maps the detected, often skewed, face onto a standardized, canonical coordinate system.
* **Result:** Every face, regardless of the original camera angle or head tilt, is **stabilized** so that key features (eyes, nose, mouth) always occupy the same relative pixel coordinates in the final $64 \times 64$ input image.

### 3.3. Smart Resolution Enhancement (Upscaling $48 \times 48 \rightarrow 64 \times 64$)

* **Option 1: Eigenface-Domain SR:**
    * The SR problem is solved in a lower-dimensional "face space" rather than the pixel domain.
    * **Benefit:** The algorithm prioritizes constructing only the **discriminative features** needed for recognition, ignoring unnecessary visual details. This maximizes feature quality within the $64 \times 64$ space and minimizes noise.
    * [[Eigenface-Domain Super-Resolution for Face Recognition]](https://www.ece.lsu.edu/ipl/papers/IEEE_TIP2003.pdf)
* **Option 2: Zero-Shot Super-Resolution (ZSSR):**
    * Used for highly heterogeneous, noisy data sources like AffectNet.
    * **Benefit:** ZSSR trains a small, dedicated CNN model **on the single input image itself** at test time. This unsupervised approach ensures the upscaling adapts perfectly to unique artifacts (like compression or sensor noise) present in that specific image, making the $64 \times 64$ feature set robust.
    * [[“Zero-Shot” Super-Resolution using Deep Internal Learning]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.pdf)

---

## 4. Strategic Data Feeding: Multi-Source Domain Adaptation (MSDA)

Directly combining and training on all source data (AffectNet and FER-2013) risks **"Negative Transfer"** because AffectNet's 'in-the-wild' images are too dissimilar from the structured FER-2013 target. Two advanced MSDA techniques are employed for safe and effective data merging:

---

### 4.1. Progressive MSDA (Similarity-Based Source Introduction)

* **Goal:** To merge diverse source domains (AffectNet) with the target domain (FER-2013) while mitigating negative transfer (where dissimilar data degrades performance).
* **Process:**
    * Start training with source samples that are **most similar** to the target domain.
    * Gradually introduce data from more **distant** source subjects (e.g., highly varied poses from AffectNet).
    * A **relevance threshold** is used to selectively filter out any source subjects that are too dissimilar, ensuring only beneficial data is integrated.
    * A **density-based memory mechanism** is maintained to prevent the model from "catastrophically forgetting" the useful samples learned in earlier, more similar domains.

* [[Progressive Feature Alignment for Unsupervised Domain Adaptation (CVPR 2019)]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Progressive_Feature_Alignment_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)

---

### 4.2. Robust Target Training ($\text{BORT}^2$) for Fine-Tuning

* **Goal:** The final, most crucial step involves refining the model on the unlabeled target domain data (i.e., the competitive test set), relying on the model's own predictions (pseudo-labels).
* **Problem:** These initial pseudo-labels are almost always noisy (incorrectly predicted). Training directly on them will compromise the final accuracy.
* **$\text{BORT}^2$ Solution (Bi-level Optimization based Robust Target Training):**
    * The fully trained MSDA model generates the noisy pseudo-labels for the target data.
    * A final target model (a specialized stochastic CNN) is trained on this pseudo-labeled data.
    * Crucially, the model is designed to measure and actively exploit **label uncertainty** (using an entropy maximization regularizer).
    * This intrinsic mechanism ensures the target model is robust against the noise in the pseudo-labels, guaranteeing the highest possible final accuracy and robustness.

* [[Robust Target Training for Multi-Source Domain Adaptation (BMVC/arXiv 2022)]](https://arxiv.org/pdf/2210.01676)


---

### 5 Takeaways

It should definitely be discussed whether the usage of different datasets is too much effort for the outcome. The models may perform better with just one data set. From a purely logical point of view, however, it would make sense to train on different data sets in order to prepare the model for the differences in later use cases, such as the live emotion classifier.

In the end, it probably makes the most sense to simply try out whether combining and using different data sets makes sense or has the desired positive effect through more data variance.

However, since the data sets are incredibly diverse in terms of the number of images and AffectNet contains by far the most images, it most probably makes the most sense to concentrate on one data set for the start.