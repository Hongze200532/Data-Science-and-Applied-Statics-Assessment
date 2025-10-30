import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.manifold import TSNE

# Load the .npz file
data = np.load('image_data.npz')

# Extract data
train_X = data['train_X']  # Training set features
train_y = data['train_Y'].flatten()  # Flatten to 1D array for Counter
test_X = data['test_X']    # Test set features
test_y = data['test_Y'].flatten()    # Flatten to 1D array (for consistency)

# --------------------------
# Question 1a Implementation
# --------------------------
print("=== Question 1a ===")
# Descriptive statistics for training and test datasets
train_items, train_features = train_X.shape
test_items, test_features = test_X.shape
print(f"Training dataset: {train_items} items, {train_features} features")
print(f"Test dataset: {test_items} items, {test_features} features")

# Class distribution in training dataset
train_class_counts = Counter(train_y)
plt.figure(figsize=(10, 6))
plt.bar(train_class_counts.keys(), train_class_counts.values())
plt.xlabel('Class Label')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Training Dataset')
plt.show()

# --------------------------
# Question 1b Implementation
# --------------------------
print("\n=== Question 1b ===")
# Assume image dimensions (e.g., 28x28 for 784 features)
img_dim = int(np.sqrt(train_features))  # Compute image dimensions from feature count
print(f"Image dimensions inferred: {img_dim}x{img_dim}")

# Select 4 random distinct classes
np.random.seed(42)
selected_classes = np.random.choice(np.unique(train_y), 4, replace=False)
print(f"Selected classes for 1b: {selected_classes}")

fig, axes = plt.subplots(4, 2, figsize=(12, 16))
for i, cls in enumerate(selected_classes):
    # Filter features for current class
    cls_features = train_X[train_y == cls]
    # Compute mean and median images
    mean_img = np.mean(cls_features, axis=0).reshape(img_dim, img_dim)
    median_img = np.median(cls_features, axis=0).reshape(img_dim, img_dim)
    # Visualize
    axes[i, 0].imshow(mean_img, cmap='gray')
    axes[i, 0].set_title(f'Mean - Class {cls}')
    axes[i, 1].imshow(median_img, cmap='gray')
    axes[i, 1].set_title(f'Median - Class {cls}')
plt.tight_layout()
plt.show()

# --------------------------
# Question 1c Implementation
# --------------------------
print("\n=== Question 1c ===")
# Select two classes for visualization
class1, class2 = selected_classes[0], selected_classes[1]
print(f"Selected classes for 1c: {class1}, {class2}")

# Filter features for the two classes
class1_feats = train_X[train_y == class1]
class2_feats = train_X[train_y == class2]

# Combine features and create labels for TSNE
combined_feats = np.vstack([class1_feats, class2_feats])
combined_labels = np.hstack([np.full(len(class1_feats), class1), np.full(len(class2_feats), class2)])

# Apply TSNE for 2D visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_embedding = tsne.fit_transform(combined_feats)

# Plot TSNE results
plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_embedding[:, 0], y=tsne_embedding[:, 1], hue=combined_labels, palette='Set1')
plt.title('TSNE Visualization of Two Classes in Training Data')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.show()
print("Discussion: If the two clusters are well-separated in the TSNE plot, it suggests potential linear separability; overlapping clusters indicate likely non-separability.")

# --------------------------
# Question 1d Implementation
# --------------------------
print("\n=== Question 1d ===")
# Choose a specific class for subset1
subset1_class = selected_classes[0]
subset_size = 50  # Define subset size

# Create subset1 (specific class)
subset1 = train_X[train_y == subset1_class][:subset_size]
# Create subset2 (random from entire training data)
np.random.seed(42)
subset2 = train_X[np.random.choice(len(train_X), subset_size, replace=False)]

# Define similarity metric (e.g., cosine similarity)
def cosine_similarity(img1, img2):
    dot = np.dot(img1, img2)
    norm1 = np.linalg.norm(img1)
    norm2 = np.linalg.norm(img2)
    return dot / (norm1 * norm2)

# Compute pairwise similarities for each subset
def compute_pairwise_sims(feats):
    sims = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            sims.append(cosine_similarity(feats[i], feats[j]))
    return sims

subset1_sims = compute_pairwise_sims(subset1)
subset2_sims = compute_pairwise_sims(subset2)

# Plot histograms
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(subset1_sims, bins=20)
plt.title(f'Similarity Distribution - Subset1 (Class {subset1_class})')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(subset2_sims, bins=20)
plt.title('Similarity Distribution - Subset2 (Random)')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
print("Explanation: Higher peak of high similarity values in subset1 histogram indicates images within the same class are more similar. Subset2’s flatter histogram shows random images are less similar.")
