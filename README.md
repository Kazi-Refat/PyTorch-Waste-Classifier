# PyTorch Waste Classifier: Organic vs. Recyclable

## Overview
[cite_start]A high-performance, binary computer vision pipeline built in PyTorch to accurately classify waste into **Organic** and **Recyclable** categories[cite: 33]. [cite_start]Engineered to handle real-world image variations, this project processes a large-scale dataset of over 25,000 images [cite: 33] using state-of-the-art deep learning techniques and hardware-accelerated training loops.

## Model Architecture & Benchmarking
This project benchmarks two prominent convolutional neural network (CNN) backbones:
* [cite_start]**ResNet-18** (Fine-tuned) [cite: 33]
* [cite_start]**Inception V3** (Fine-tuned) [cite: 33]

## Advanced Training Strategies
To ensure robust feature extraction and prevent overfitting, the training pipeline incorporates several advanced optimizations:
* [cite_start]**Class-Weighted Loss:** Implemented to penalize majority-class biases and handle dataset imbalances effectively[cite: 33].
* [cite_start]**Optimization & Scheduling:** Utilized the **AdamW** optimizer [cite: 34] [cite_start]paired with a **Cosine Annealing Learning Rate scheduler** to achieve smooth, optimal convergence[cite: 33]. 
* [cite_start]**Regularization:** Integrated **early stopping** mechanisms to halt training at peak validation performance[cite: 33].
* [cite_start]**Hardware Acceleration:** Significantly accelerated the training loops using **CUDA Mixed Precision** (`torch.amp.autocast`), optimizing VRAM usage while maintaining numerical stability[cite: 34].

## Performance Metrics
The fine-tuned **ResNet-18** architecture achieved the strongest results across the benchmarked models, delivering:
* [cite_start]**Validation Accuracy:** 93.8% [cite: 34]
* [cite_start]**Precision:** 95.1% [cite: 34]
* [cite_start]**Recall:** 90.7% [cite: 34]

## Tech Stack
* [cite_start]**Core Language:** Python [cite: 40]
* [cite_start]**Deep Learning Framework:** PyTorch [cite: 40]
* [cite_start]**Hardware Optimization:** CUDA (`torch.amp.autocast`) [cite: 34]

---
### ⚙️ Quick Start & Installation
*(Add your installation steps here, e.g., `pip install -r requirements.txt`)*

### 🚀 Usage
*(Add a quick code snippet on how to run the inference script or train the model here)*
