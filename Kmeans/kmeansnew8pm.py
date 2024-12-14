from sklearn.cluster import KMeans
import numpy as np
import json
from sklearn.metrics import silhouette_score

# Define file paths
input_pca_matrix_path = r"C:\Users\A1157\Downloads\Lab240123final\new_pca_tfidf_matrix.npy"
output_clusters_path = r"C:\Users\A1157\Downloads\Lab240123final\kmeans_clusters.json"
output_feature_names_path = r"C:\Users\A1157\Downloads\Lab240123final\tfidf_features_sorted.json"
output_cluster_centers_path = r"C:\Users\A1157\Downloads\Lab240123final\kmeans_cluster_centers.npy"
output_textual_clusters_path = r"C:\Users\A1157\Downloads\Lab240123final\kmeans_textual_clusters.json"


# Load the PCA-reduced TF-IDF matrix
pca_tfidf_matrix = np.load(input_pca_matrix_path)

n_clusters = 5  # Adjusted based on the elbow method
print(f"Clustering using K-means with {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(pca_tfidf_matrix)

# Save the cluster labels
clusters_dict = {"cluster_labels": cluster_labels.tolist()}
with open(output_clusters_path, 'w', encoding='utf-8') as f:
    json.dump(clusters_dict, f, indent=4)

print(f"K-means clustering results saved to {output_clusters_path}")

# Save the cluster centers to a file
np.save(output_cluster_centers_path, kmeans.cluster_centers_)
print(f"K-means cluster centers saved to {output_cluster_centers_path}")

print("Extracting top features for each cluster...")
with open(output_feature_names_path, 'r', encoding='utf-8') as f:
    feature_names = json.load(f)

top_features_per_cluster = 10  # Number of top features per cluster to display
top_features_dict = {}

for i, cluster_center in enumerate(kmeans.cluster_centers_):
    # Get the top features for the cluster
    top_feature_indices = cluster_center.argsort()[-top_features_per_cluster:][::-1]
    top_terms = [feature_names[index] for index in top_feature_indices]
    top_features_dict[f"Cluster_{i+1}"] = top_terms
    print(f"Cluster {i+1} key terms: {top_terms}")

# Save the human-readable cluster features to a JSON file
with open(output_textual_clusters_path, 'w', encoding='utf-8') as f:
    json.dump(top_features_dict, f, indent=4)

print(f"Top features per cluster saved to {output_textual_clusters_path}")