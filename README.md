# Financial Fraud Classification using Graph Neural Networks (GNN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/financial-fraud-classification/blob/main/notebooks/gnn_fraud_detection_baseline.ipynb)

## Overview

This project implements a deep learning approach to detect fraudulent credit card transactions using **Graph Neural Networks (GNNs)**. Traditional fraud detection models analyze transactions independently, but this project models transactions as a **graph of relationships**, enabling the model to learn patterns of similarity between transactions.

A **Graph Convolutional Network (GCN)** is implemented using **PyTorch** and **PyTorch Geometric**. Transactions are represented as nodes in a graph, and edges are created between similar transactions using a **K-Nearest Neighbors (KNN)** approach. This allows the model to learn from both transaction features and the structure of relationships between transactions.

---

## Technologies Used

* PyTorch
* PyTorch Geometric
* Scikit-learn
* NumPy
* Matplotlib
* Google Colab

---

## Dataset

The model is trained on the **Credit Card Fraud Detection dataset** from Kaggle:

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains anonymized credit card transactions made by European cardholders and is **highly imbalanced**, with fraudulent transactions representing only a tiny fraction of the total data.

To manage this imbalance and keep the graph computationally manageable, legitimate transactions were **under-sampled** to maintain a **5:1 ratio of legitimate to fraudulent transactions**.

---

## Methodology

### 1. Data Preprocessing

All features were standardized (mean = 0, variance = 1) to ensure effective neural network training.

### 2. Graph Construction

A **K-Nearest Neighbors (KNN) graph** with (k = 5) was constructed to connect transactions with similar feature representations.

### 3. Model Architecture

A custom **Graph Neural Network** was implemented using two **GCNConv** layers:

* Graph Convolution Layer
* ReLU activation
* Graph Convolution Layer
* Log-Softmax output

### 4. Training

Transaction nodes were split into **80% training** and **20% testing** sets using **stratified sampling** to preserve the fraud ratio.

The model was trained for **50 epochs** using:

* **Adam optimizer**
* **Cross-Entropy Loss**

---

## Results

Fraud detection systems must balance two competing goals:

* **Precision** → minimizing false fraud alerts
* **Recall** → successfully detecting fraudulent activity

The trained Graph Neural Network achieved strong performance across these critical metrics.

### Final Test Set Metrics

| Metric    | Score      |
| --------- | ---------- |
| Accuracy  | **0.9746** |
| Precision | **0.9770** |
| Recall    | **0.8673** |
| F1 Score  | **0.9189** |
| AUC-ROC   | **0.9746** |

These results demonstrate that modeling **transaction relationships with graph neural networks** can significantly improve fraud detection performance.

---

## Repository Structure

```
financial-fraud-classification
│
├── notebooks
│   └── gnn_fraud_detection_baseline.ipynb
│
├── models
│   └── gnn_fraud_model.pth
│
└── README.md
```

* **notebooks/** – Jupyter notebook containing the full ML pipeline
* **models/** – Saved PyTorch model checkpoint
* **README.md** – Project documentation

---

## Getting Started

To run this project:

1. Open the notebook in **Google Colab** using the badge above.
2. Download `creditcard.csv` from Kaggle.
3. Upload the dataset to your Colab environment or mount it via Google Drive.
4. Run the notebook cells sequentially to:

   * install dependencies
   * construct the transaction graph
   * train the Graph Neural Network
   * evaluate model performance

---

## Model Checkpoint

The repository includes a pretrained baseline model:

```
models/gnn_fraud_model.pth
```

This file contains the **trained PyTorch model weights**, allowing the model to be loaded without retraining.

---

## Future Improvements

Potential improvements to this project include:

* experimenting with **Graph Attention Networks (GAT)**
* exploring **GraphSAGE architectures**
* testing different **graph construction strategies**
* performing **hyperparameter tuning**
* evaluating performance on **larger graph samples**

---
