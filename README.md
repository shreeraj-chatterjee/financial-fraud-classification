# Financial Fraud Detection using Graph Neural Networks (GNN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/financial-fraud-classification/blob/main/notebooks/gat_focal_loss_fraud_model.ipynb)

> **Why This Project Matters** - Fraud detection is fundamentally relational, not independent. A credit card transaction doesn't happen in a vacuum. This project demonstrates how graph-based deep learning can capture complex relationships and subtle anomalies at scale, outperforming traditional tabular-only machine learning models.

## Overview

This project builds a production-oriented fraud detection system using **Graph Neural Networks (GNNs)**. Unlike traditional models that treat transactions independently, this approach models transactions as a graph of relationships, enabling the system to detect subtle fraud patterns through connectivity and feature similarity. 

**Final Result:** A scalable GAT + FAISS + Focal Loss pipeline trained on a massive, real-world, highly imbalanced dataset (~577:1 ratio).

## Key Features

* **Graph-Based Detection:** Moves beyond standard tabular ML to leverage network structures.
* **Scalable Graph Construction:** Utilizes FAISS for lightning-fast similarity search on hundreds of thousands of nodes.
* **Extreme Imbalance Handling:** Replaces static class weights with Focal Loss to dynamically focus on rare fraud cases.
* **Progressive Development:** Documents the engineering journey from a basic GCN to a production-ready GAT pipeline.
* **End-to-End ML Workflow:** Complete pipeline from data preprocessing to graph construction, model training, and evaluation.

## Tech Stack

* **Deep Learning:** PyTorch, PyTorch Geometric (PyG)
* **Vector Search:** FAISS (CPU/GPU-accelerated)
* **Data Processing:** Scikit-learn, NumPy, Pandas
* **Visualization:** Matplotlib
* **Environment:** Google Colab / Jupyter

## Dataset

* **Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Total Transactions:** 284,315
* **Fraud Cases:** 492
* **Imbalance Ratio:** ~577:1

## Methodology

The project evolves across four distinct engineering stages:

**1. Baseline (GCN + Undersampling)**
* Built the initial graph pipeline using a 2-layer Graph Convolutional Network.
* *Limitation:* Reduced dataset imbalance to an artificial 5:1 ratio to fit memory constraints, resulting in an unrealistic distribution.

**2. Regularization (Dropout)**
* Introduced Dropout layers to reduce mild overfitting.
* *Result:* Improved model generalization and recall scores.

**3. Architectural Upgrade (Graph Attention Network - GAT)**
* Upgraded from GCN to GAT, introducing multi-head attention.
* *Result:* Allowed the model to weigh the importance of specific neighbors, vastly improving precision by focusing strictly on suspicious connections.

**4. Production Model (FAISS + Focal Loss)**
* **FAISS:** Replaced `sklearn` nearest-neighbors, reducing graph construction time on the full dataset from ~40 minutes to seconds.
* **Focal Loss:** Shifted learning focus to rare fraud cases without being overwhelmed by false positives.
* *Result:* Successfully trained on the full, real-world 577:1 data distribution.

## Results (Final Model)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.9992 |
| **Precision** | 0.7404 |
| **Recall** | 0.7857 |
| **F1 Score** | 0.7624 |
| **AUC-ROC** | 0.9615 |

*Note: Earlier baseline models showed artificially inflated F1 scores (>0.90) due to 5:1 undersampling. The final model reflects highly robust, real-world performance under extreme imbalance.*

## Quick Start

```bash
# Clone repo
git clone [https://github.com/YOUR_USERNAME/financial-fraud-classification.git](https://github.com/YOUR_USERNAME/financial-fraud-classification.git)
cd financial-fraud-classification

# Install dependencies
pip install -r requirements.txt
```

Then, launch the final notebook:
```bash
jupyter notebook notebooks/gat_focal_loss_fraud_model.ipynb
```
*Alternatively, use the "Open in Colab" badge at the top of this repository.*

## Repository Structure

```text
financial-fraud-classification/
│
├── notebooks/
│   ├── gnn_fraud_detection_baseline.ipynb
│   ├── gnn_fraud_detection_dropout.ipynb
│   ├── gat_fraud_detection.ipynb
│   └── gat_focal_loss_fraud_model.ipynb
│
├── models/
│   ├── gnn_fraud_model.pth
│   ├── gnn_dropout.pth
│   ├── gat_fraud_model.pth
│   └── gat_focal_loss.pth
│
├── requirements.txt
├── LICENSE
└── README.md
```

## Future Work

* Explore **GraphSAGE** architectures for inductive learning (classifying entirely new, unseen transactions without rebuilding the whole graph).
* Test alternative graph construction strategies (e.g., radius/epsilon-neighborhoods instead of strictly KNN).
* Perform exhaustive hyperparameter optimization (tuning attention heads, hidden dimensions, and Focal Loss parameters).
