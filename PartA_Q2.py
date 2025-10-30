import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, adjusted_rand_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')


# Load data and flatten labels
data = np.load('image_data.npz')
train_X = data['train_X']
train_y = data['train_Y'].flatten()
test_X = data['test_X']
test_y = data['test_Y'].flatten()

# --------------------------
# Prepare two-class subsets 
# --------------------------
class_counts = Counter(train_y)
sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
target_classes = [cls for cls, _ in sorted_classes[:2]]
print(f"Selected classes: {target_classes}")

# Balance training data
train_mask = np.isin(train_y, target_classes)
train_subset = train_X[train_mask]
train_labels = train_y[train_mask]
class_counts = Counter(train_labels)
min_train = min(class_counts.values())
train_balanced = np.vstack([
    train_subset[train_labels == target_classes[0]][:min_train],
    train_subset[train_labels == target_classes[1]][:min_train]
])
train_labels_balanced = np.hstack([
    np.full(min_train, target_classes[0]),
    np.full(min_train, target_classes[1])
])

# Balance test data
test_mask = np.isin(test_y, target_classes)
test_subset = test_X[test_mask]
test_labels = test_y[test_mask]
class_counts_test = Counter(test_labels)
min_test = min(class_counts_test.values())
test_balanced = np.vstack([
    test_subset[test_labels == target_classes[0]][:min_test],
    test_subset[test_labels == target_classes[1]][:min_test]
])
test_labels_balanced = np.hstack([
    np.full(min_test, target_classes[0]),
    np.full(min_test, target_classes[1])
])

# Binary labels (0/1)
binary_train_labels = np.where(train_labels_balanced == target_classes[0], 0, 1)
binary_test_labels = np.where(test_labels_balanced == target_classes[0], 0, 1)

# Standardize features
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_balanced)
test_scaled = scaler.transform(test_balanced)

# --------------------------
# 2a(i): Basic models
# --------------------------
def evaluate(y_true, y_pred, model_name):
    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=1):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=1):.4f}")
    print(f"F1: {f1_score(y_true, y_pred, zero_division=1):.4f}")

# Logistic Regression (basic)
lr_basic = LogisticRegression(max_iter=1000, random_state=42)
lr_basic.fit(train_scaled, binary_train_labels)
evaluate(binary_test_labels, lr_basic.predict(test_scaled), "Logistic Regression (Basic)")

# KNN (basic)
knn_basic = KNeighborsClassifier()
knn_basic.fit(train_scaled, binary_train_labels)
evaluate(binary_test_labels, knn_basic.predict(test_scaled), "KNN (Basic)")

# Random Forest (basic)
rf_basic = RandomForestClassifier(random_state=42)
rf_basic.fit(train_scaled, binary_train_labels)
evaluate(binary_test_labels, rf_basic.predict(test_scaled), "Random Forest (Basic)")

# --------------------------
# 2a(ii): Hyperparameter tuning with balanced K-fold CV
# --------------------------
def k_fold_cv(X, y, model, params, k=5):
    fold_size = len(X) // k
    best_score = -1
    best_param = None
    
    for param in params:
        scores = []
        for i in range(k):
            val_indices = []
            for cls in [0, 1]:
                cls_indices = np.where(y == cls)[0]
                start = i * (len(cls_indices) // k)
                end = (i + 1) * (len(cls_indices) // k)
                val_indices.extend(cls_indices[start:end])
            
            X_val = X[val_indices]
            y_val = y[val_indices]
            train_indices = [idx for idx in range(len(X)) if idx not in val_indices]
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            
            model.set_params(**param)
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val)
            scores.append(f1_score(y_val, y_pred, zero_division=1))
        
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_param = param
    
    return best_param, best_score

# Tune Logistic Regression (C)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr_params = [{'C': 0.001}, {'C': 0.01}, {'C': 0.1}, {'C': 1}, {'C': 10}]
best_lr_param, best_lr_score = k_fold_cv(train_scaled, binary_train_labels, lr, lr_params)
print(f"\nBest Logistic Regression Param: {best_lr_param} (CV F1: {best_lr_score:.4f})")
evaluate(binary_test_labels, LogisticRegression(** best_lr_param, max_iter=1000).fit(train_scaled, binary_train_labels).predict(test_scaled), "Logistic Regression (Tuned)")

# Tune KNN (n_neighbors)
knn = KNeighborsClassifier()
knn_params = [{'n_neighbors': 1}, {'n_neighbors': 3}, {'n_neighbors': 5}, {'n_neighbors': 7}, {'n_neighbors': 9}]
best_knn_param, best_knn_score = k_fold_cv(train_scaled, binary_train_labels, knn, knn_params)
print(f"\nBest KNN Param: {best_knn_param} (CV F1: {best_knn_score:.4f})")
evaluate(binary_test_labels, KNeighborsClassifier(**best_knn_param).fit(train_scaled, binary_train_labels).predict(test_scaled), "KNN (Tuned)")

# Tune Random Forest (n_estimators)
rf = RandomForestClassifier(random_state=42)
rf_params = [{'n_estimators': 50}, {'n_estimators': 100}, {'n_estimators': 200}]
best_rf_param, best_rf_score = k_fold_cv(train_scaled, binary_train_labels, rf, rf_params)
print(f"\nBest Random Forest Param: {best_rf_param} (CV F1: {best_rf_score:.4f})")
evaluate(binary_test_labels, RandomForestClassifier(** best_rf_param, random_state=42).fit(train_scaled, binary_train_labels).predict(test_scaled), "Random Forest (Tuned)")

# --------------------------
# 2b: K-means clustering with proper evaluation
# --------------------------
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(test_scaled)

def adjust_cluster_labels(true_labels, cluster_labels):
    cm = confusion_matrix(true_labels, cluster_labels)
    if cm[0,0] + cm[1,1] < cm[0,1] + cm[1,0]:
        return 1 - cluster_labels
    return cluster_labels

adjusted_clusters = adjust_cluster_labels(binary_test_labels, cluster_labels)

ari = adjusted_rand_score(binary_test_labels, adjusted_clusters)
print(f"\nK-means Adjusted Rand Index: {ari:.4f}")
print("Confusion Matrix (True vs Adjusted Clusters):")
print(confusion_matrix(binary_test_labels, adjusted_clusters))

pca = PCA(n_components=2)
test_pca = pca.fit_transform(test_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

selector = SelectKBest(score_func=f_classif, k=2)
train_selected = selector.fit_transform(train_scaled, binary_train_labels)
test_selected = selector.transform(test_scaled)


top_feat_indices = np.argsort(selector.scores_)[-2:] 


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(test_selected[:,0], test_selected[:,1], c=binary_test_labels, cmap='coolwarm', alpha=0.6)
plt.title('True Labels (Top 2 Features)')


kmeans_feat = KMeans(n_clusters=2, n_init=10, random_state=42)
cluster_labels_feat = kmeans_feat.fit_predict(test_selected)
adjusted_clusters_feat = adjust_cluster_labels(binary_test_labels, cluster_labels_feat)

plt.subplot(1, 2, 2)
plt.scatter(test_selected[:,0], test_selected[:,1], c=adjusted_clusters_feat, cmap='viridis', alpha=0.6)
plt.scatter(kmeans_feat.cluster_centers_[:,0], kmeans_feat.cluster_centers_[:,1], marker='X', s=200, c='red', label='Centers')
plt.title('Adjusted K-means Clusters (Top 2 Features)')
plt.legend()
plt.tight_layout()
plt.show()
