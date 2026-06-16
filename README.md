# PyTorch Waste Classifier: Organic vs. Recyclable

## Overview
A high-performance, binary computer vision pipeline built in PyTorch to accurately classify waste into **Organic** and **Recyclable** categories. Engineered to handle real-world image variations, this project processes a large-scale dataset of over 25,000 images using state-of-the-art deep learning techniques and hardware-accelerated training loops.

## Model Architecture & Benchmarking
This project benchmarks two prominent convolutional neural network (CNN) backbones:
* **ResNet-18** (Fine-tuned)
* **Inception V3** (Fine-tuned)

## Advanced Training Strategies
To ensure robust feature extraction and prevent overfitting, the training pipeline incorporates several advanced optimizations:
* **Class-Weighted Loss:** Implemented to penalize majority-class biases and handle dataset imbalances effectively.
* **Optimization & Scheduling:** Utilized the **AdamW** optimizer paired with a **Cosine Annealing Learning Rate scheduler** to achieve smooth, optimal convergence. 
* **Regularization:** Integrated **early stopping** mechanisms to halt training at peak validation performance.
* **Hardware Acceleration:** Significantly accelerated the training loops using **CUDA Mixed Precision** (`torch.amp.autocast`), optimizing VRAM usage while maintaining numerical stability.

## Performance Metrics
The fine-tuned **ResNet-18** architecture achieved the strongest results across the benchmarked models, delivering:
* **Validation Accuracy:** 93.8%
* **Precision:** 95.1%
* **Recall:** 90.7%

## Tech Stack
* **Core Language:** Python
* **Deep Learning Framework:** PyTorch
* **Hardware Optimization:** CUDA (`torch.amp.autocast`)

---
### ⚙️ Quick Start & Installation
*(Add your installation steps here, e.g., `pip install -r requirements.txt`)*

### 🚀 Usage
*(Add a quick code snippet on how to run the inference script or train the model here)*
