### Classification Model Results

---

#### Logistic Regression (Basic)

- **Accuracy:** 1.0000  
- **Precision:** 1.0000  
- **Recall:** 1.0000  
- **F1-score:** 1.0000  

---

#### KNN (Basic)

- **Accuracy:** 1.0000  
- **Precision:** 1.0000  
- **Recall:** 1.0000  
- **F1-score:** 1.0000  

---

#### Random Forest (Basic)

- **Accuracy:** 1.0000  
- **Precision:** 1.0000  
- **Recall:** 1.0000  
- **F1-score:** 1.0000  

---

### Tuned Models and Hyperparameters

#### Logistic Regression (Tuned)

- **Best hyperparameter (from cross-validation):**  
  - `C = 0.01`  
  - **Best CV F1-score:** 0.9810  
- **Test performance after tuning:**  
  - **Accuracy:** 1.0000  
  - **Precision:** 1.0000  
  - **Recall:** 1.0000  
  - **F1-score:** 1.0000  

---

#### KNN (Tuned)

- **Best hyperparameter (from cross-validation):**  
  - `n_neighbors = 1`  
  - **Best CV F1-score:** 0.9845  
- **Test performance after tuning:**  
  - **Accuracy:** 1.0000  
  - **Precision:** 1.0000  
  - **Recall:** 1.0000  
  - **F1-score:** 1.0000  

---

#### Random Forest (Tuned)

- **Best hyperparameter (from cross-validation):**  
  - `n_estimators = 50`  
  - **Best CV F1-score:** 0.9828  
- **Test performance after tuning:**  
  - **Accuracy:** 1.0000  
  - **Precision:** 1.0000  
  - **Recall:** 1.0000  
  - **F1-score:** 1.0000  

---

### Clustering Evaluation (K-means)

- **Adjusted Rand Index (ARI):** 0.0000  

The ARI of **0.0000** indicates that the K-means clustering assignments are no better than random with respect to the true labels.

**Confusion Matrix (True Labels vs. Adjusted Clusters):**

| True \ Predicted Cluster | Cluster 0 | Cluster 1 |
|--------------------------|----------:|----------:|
| **Class 0**              | 34        | 0         |
| **Class 1**              | 33        | 1         |

This confusion matrix shows that most samples are assigned to a single cluster, leading to poor alignment with the true class labels.
