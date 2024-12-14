from sklearn.decomposition import TruncatedSVD
from scipy.sparse import load_npz
import numpy as np
import json

# Define file paths
input_tfidf_matrix_path = r"C:\Users\A1157\Downloads\Lab240123final\tfidf_matrix.npz"
output_pca_matrix_path = r"C:\Users\A1157\Downloads\Lab240123final\new_pca_tfidf_matrix.npy"
output_feature_names_path = r"C:\Users\A1157\Downloads\Lab240123final\tfidf_features_sorted.json"
output_top_features_path = r"C:\Users\A1157\Downloads\Lab240123final\new_top_features_per_component.json"

# Load the sparse TF-IDF matrix
tfidf_matrix = load_npz(input_tfidf_matrix_path)

# Initialize PCA (using TruncatedSVD as PCA for sparse data)
n_components = 1000  # Adjust based on balance of interpretability and efficiency
pca = TruncatedSVD(n_components=n_components)

# Fit and transform the TF-IDF matrix with PCA
pca_tfidf_matrix = pca.fit_transform(tfidf_matrix)

# Save the reduced PCA matrix
np.save(output_pca_matrix_path, pca_tfidf_matrix)

# Optional: Explained variance to understand data retention
explained_variance = np.sum(pca.explained_variance_ratio_)
print(f"Total explained variance with {n_components} components: {explained_variance:.2f}")

print(f"PCA-reduced TF-IDF matrix saved to {output_pca_matrix_path}")

# Load the feature names
with open(output_feature_names_path, 'r', encoding='utf-8') as f:
    feature_names = json.load(f)

# Get the top contributing features for each component
top_features_per_component = 10  # Number of features per component to display
top_features_dict = {}

for i in range(n_components):
    component = pca.components_[i]
    top_feature_indices = component.argsort()[-top_features_per_component:][::-1]
    top_terms = [feature_names[index] for index in top_feature_indices]
    top_features_dict[f"Component_{i+1}"] = top_terms
    print(f"Top terms in Component {i+1}: {top_terms}")

# Save the top contributing features to a JSON file
with open(output_top_features_path, 'w', encoding='utf-8') as f:
    json.dump(top_features_dict, f, indent=4)

print(f"Top contributing features per component saved to {output_top_features_path}")


