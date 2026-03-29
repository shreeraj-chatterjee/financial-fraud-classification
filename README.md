# Financial Fraud Detection using Graph Neural Networks (GNNs)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shreeraj-chatterjee/financial-fraud-classification/blob/main/notebooks/graphsage_mini_batch.ipynb)

> **Why This Project Matters** - Fraud detection is fundamentally relational, not independent. A credit card transaction doesn't happen in a vacuum. This project demonstrates how graph-based deep learning can capture complex relationships and subtle anomalies at scale, outperforming traditional tabular-only machine learning models.

## Overview

This project builds a production-oriented fraud detection system using **Graph Neural Networks (GNNs)**. Unlike traditional models that treat transactions independently, this approach models transactions as a graph of relationships, enabling the system to detect subtle fraud patterns through connectivity and feature similarity. 

**Final Result:** A scalable GraphSAGE + FAISS + Focal Loss pipeline trained on a massive, real-world, highly imbalanced dataset (~577:1 ratio).

## Progressive Development

This project was built iteratively, starting from a basic baseline and evolving into a scalable, production-ready architecture. 

1. **GCN Baseline (`gnn_fraud_detection_baseline.ipynb`):** A standard Graph Convolutional Network. Used 5:1 undersampling to fit memory constraints, resulting in artificially high but unrealistic performance.
2. **GCN + Dropout (`gnn_fraud_detection_dropout.ipynb`):** Added regularization to prevent the model from overfitting.
3. **GAT Architecture (`gat_fraud_detection.ipynb`):** Introduced Graph Attention Networks, allowing the model to weigh the importance of different neighboring transactions dynamically. Evaluated on the full dataset.
4. **GAT + Focal Loss (`gat_focal_loss_fraud_model.ipynb`):** Implemented Focal Loss to mathematically force the model to focus on the hard-to-classify fraudulent nodes, drastically improving Recall. Graph construction was isolated to prevent data leakage.
5. **GraphSAGE (`graphsage_fraud_model.ipynb`):** Transitioned from transductive to **inductive learning**, allowing the model to generate embeddings for entirely unseen transactions without rebuilding the entire graph.
6. **GraphSAGE + Mini-Batching (`graphsage_mini_batch.ipynb`):** Implemented PyTorch Geometric's `NeighborLoader` to enable mini-batch training. This makes the architecture scalable to massive, out-of-core production datasets.
7. **GraphSAGE + Radius Search (`graphsage_radius_search.ipynb`):** Replaced KNN graph construction with a FAISS-powered **Capped Radius (Epsilon) Search**. This prevents the artificial connection of isolated anomalous nodes, resulting in the highest AUC-ROC of the project.

## Results Summary

*Note: All models from the GAT+Focal Loss stage onward utilize a strict Train/Validation/Test split to prevent data leakage, with threshold tuning performed exclusively on the Validation set.*

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| GCN Baseline | - | - | - | >0.90 * | - |
| GCN + Dropout | - | - | - | >0.90 * | - |
| GAT | 99.85% | 0.66 | 0.17 | 0.27 | 0.81 |
| GAT + Focal Loss | 99.94% | 0.90 | 0.75 | 0.8235 | 0.87 |
| GraphSAGE (Inductive) | 99.94% | 0.90 | 0.75 | 0.8235 | 0.88 |
| **GraphSAGE (Mini-Batch)** | **99.95%** | **0.90** | **0.77** | **0.8321** | 0.94 |
| **GraphSAGE (Radius)** | 99.94% | 0.85 | 0.77 | 0.8085 | **0.96** |

*\* The early GCN models were evaluated on a heavily undersampled dataset (5:1 ratio), artificially inflating the F1 score. All subsequent models were evaluated on the true, real-world 577:1 distribution to reflect actual production viability.*

### Key Takeaways
* **Best Business KPI:** The **GraphSAGE (Mini-Batch)** model achieved the highest F1-Score (0.8321). In a production setting, this configuration currently offers the best mathematical trade-off between catching fraud and minimizing false positives for legitimate customers.
* **Best Structural Separability:** The **GraphSAGE (Radius Search)** model achieved an exceptional AUC-ROC of 0.9615. By only connecting transactions within a strict feature distance, it successfully isolated anomalous fraud patterns, proving to be the most structurally sound approach.

## Tech Stack
* **Deep Learning:** PyTorch, PyTorch Geometric (PyG)
* **Graph Construction:** FAISS (Facebook AI Similarity Search)
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib

## Project Roadmap & Future Work
- [x] Build initial graph pipeline using a 2-layer Graph Convolutional Network (GCN).
- [x] Introduce Dropout layers to the GCN to reduce mild overfitting.
- [x] Upgrade architecture to Graph Attention Network (GAT) to leverage multi-head attention.
- [x] Implement Focal Loss and FAISS to handle extreme dataset imbalance and scale graph construction without data leakage.
- [x] Explore **GraphSAGE** architectures for inductive learning (classifying entirely new, unseen transactions).
- [x] Test alternative graph construction strategies (e.g., radius/epsilon-neighborhoods instead of strictly KNN).
- [ ] Perform exhaustive **Hyperparameter Optimization** (using Optuna to tune hidden dimensions, dropout rates, and Focal Loss alpha/gamma parameters to maximize F1 on the Radius Search model).
- [ ] Integrate tabular and graph-based predictions using an ensemble approach.
