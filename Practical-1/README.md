# Research Paper Implementation: Snow Detection for Pedestrian Safety

Implementation and replication study for **Generative AI / Deep Learning Lab Assignment 1**.

---

## 1. Research Paper Overview

* **Paper Title:** *Image Classification for Snow Detection to Improve Pedestrian Safety*[cite: 2]
* **Authors:** Ricardo de Deijn, Rajeev Bukralia (Minnesota State University, Mankato)[cite: 2]
* **Conference:** Proceedings of the 19th Midwest Association for Information Systems Conference (MWAIS 2024)[cite: 2]
* **Paper Link:** [arXiv:2407.00818](https://arxiv.org/abs/2407.00818)[cite: 2]
* **Summary:** This paper proposes a computer vision transfer learning approach to detect snow-covered pavements using smartphone imagery[cite: 2]. Designed to prevent winter fall injuries among vulnerable demographics (elderly and visually impaired individuals), the study benchmarks pre-trained **ResNet-50** and **VGG-19** CNNs fine-tuned on a custom dataset[cite: 2].

---

## 2. Dataset Description

* **Dataset Name:** `Fitda/Snowy_Sidewalk_Detection`
* **Hugging Face Link:** [Fitda/Snowy_Sidewalk_Detection](https://huggingface.co/datasets/Fitda/Snowy_Sidewalk_Detection)
* **Total Samples:** 118 images (Smartphone camera captures in Minnesota, USA)[cite: 2]
  * **Train Split:** 78 samples (paired locations)[cite: 2]
  * **Validation Split:** 18 samples (paired locations)[cite: 2]
  * **Test Split:** 22 samples (distinct, unseen locations)[cite: 2]
* **Classes:** Binary Classification — `0: Snow-Free`, `1: Snow`[cite: 2]

---

## 3. Methodology & Implementation Pipeline

### Task 1: Preprocessing[cite: 1]
* **Resizing:** Scaled raw high-resolution images down to $128 \times 128$ pixels[cite: 2].
* **Normalization:** Scaled pixel values to $[0, 1]$ and normalized using standard ImageNet channel statistics[cite: 2]:
  * Mean: `[0.485, 0.456, 0.406]`[cite: 2]
  * Std: `[0.229, 0.224, 0.225]`[cite: 2]
* **DataLoader Pipeline:** Applied on-the-fly PyTorch batch transforms with a custom `collate_fn` and batch size of 4[cite: 2].

### Task 2: Model Architecture & Fine-Tuning[cite: 1]
* **Base Backbone:** Pre-trained `ResNet-50` (ImageNet weights)[cite: 2].
* **Layer Freezing:** All convolutional feature extraction layers were frozen (`requires_grad = False`) to prevent overfitting on the small dataset[cite: 1, 2].
* **Classifier Replacement:** Replaced the final Linear layer (`model.fc`) with an output dimension of 2 classes[cite: 2].
* **Hyperparameters:**
  * Optimizer: `Adam`[cite: 2]
  * Learning Rate: `0.0001`[cite: 2]
  * Epochs: `15`[cite: 2]
  * Loss Function: Cross-Entropy Loss (`nn.CrossEntropyLoss`)[cite: 2]
* **Feature Map Visualizations:** Extracted intermediate activations via PyTorch forward hooks on `conv1` (low-level edges) and `layer2` (mid-level textures)[cite: 1].

---

## 4. Experimental Results & Performance Comparison

### Model Performance Metrics (Test Set = 22 Images)[cite: 2]

| Metric | Our Implementation (ResNet-50) | Paper's Individual ResNet-50[cite: 2] | Paper's Best Ensemble Model[cite: 2] |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **77.27%** | 54.5%[cite: 2] | **72.7%**[cite: 2] |
| **F1-Score** | **73.68%** | 50.9%[cite: 2] | **71.8%**[cite: 2] |
| **Precision** | **87.50%** | Not Reported | Not Reported |
| **Recall** | **63.64%** | Not Reported | Not Reported |

### Confusion Matrix Breakdown (Test Set)
* **True Negatives (Snow-Free correctly identified):** 10
* **False Positives (Snow-Free misclassified as Snow):** 1
* **True Positives (Snow correctly identified):** 7
* **False Negatives (Snow missed, misclassified as Snow-Free):** 4

---

## 5. Key Findings & Discussion

1. **Baseline Outperformance:** Our fine-tuned ResNet-50 model achieved **77.27% accuracy** and **73.68% F1-score**, exceeding both the paper's standalone ResNet-50 baseline ($54.5\%$) and their top ensemble benchmark ($72.7\%$)[cite: 2].
2. **Safety-Critical Analysis:** The model exhibited high Precision ($87.50\%$) but moderate Recall ($63.64\%$). In pedestrian safety systems, **False Negatives (4 cases)** represent the highest risk, as failing to alert a visually impaired user about snowy/icy sidewalk hazards can cause fall injuries[cite: 2].
3. **Future Improvements:**
   * **Recall-Weighted / Focal Loss:** Implement class-weighted penalty functions to prioritize minimizing False Negatives[cite: 2].
   * **Data Augmentation:** Apply random brightness jitter, contrast variation, and motion blur to generalize against sunlight reflections on wet pavement[cite: 1, 2].
   * **Multi-Modal Integration:** Fuse ambient environmental parameters (temperature, humidity) directly into the classification head[cite: 2].

---

## 6. How to Run

1. Open `Practical_Assignment_1_Submission.ipynb` in **Google Colab**[cite: 1].
2. Set Runtime to **GPU (T4)**.
3. Install required libraries:
   ```bash
   pip install datasets torchvision seaborn scikit-learn