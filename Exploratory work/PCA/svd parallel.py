from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, vstack
from joblib import Parallel, delayed
import json
import numpy as np
import os

# Define file paths for Midway
cleaned_data_path = "/path/to/fully_cleaned_data.json"  # Update with the actual path on Midway
output_tfidf_matrix_path = "/path/to/tfidf_matrix.npz"  # Update with the actual path on Midway
output_feature_names_path = "/path/to/tfidf_features_sorted.json"  # Update with the actual path on Midway

def load_data_in_chunks(filepath, chunk_size=10000):
    """Load data from a JSON file in chunks."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Yield chunks of data
    for i in range(0, len(data), chunk_size):
        yield [" ".join(post) for post in data[i:i + chunk_size]]

def process_chunk_with_tfidf(chunk, vectorizer=None):
    """Process a single chunk with TF-IDF, fitting the vectorizer on the first chunk."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(chunk)
    else:
        tfidf_matrix = vectorizer.transform(chunk)
    return tfidf_matrix, vectorizer

def main():
    # Initialize the shared TF-IDF vectorizer and storage for matrices
    vectorizer = TfidfVectorizer()
    tfidf_matrices = []

    # Load and process each chunk in parallel
    with Parallel(n_jobs=-1, backend="multiprocessing") as parallel:
        tfidf_results = parallel(
            delayed(process_chunk_with_tfidf)(chunk, vectorizer)
            for chunk in load_data_in_chunks(cleaned_data_path)
        )

    # Collect results and combine matrices
    for tfidf_matrix, _ in tfidf_results:
        tfidf_matrices.append(tfidf_matrix)
    
    combined_tfidf_matrix = vstack(tfidf_matrices)
    save_npz(output_tfidf_matrix_path, combined_tfidf_matrix)
    print(f"TF-IDF matrix saved to {output_tfidf_matrix_path}")

    # Calculate and save sorted feature names
    feature_names = vectorizer.get_feature_names_out()
    feature_sums = combined_tfidf_matrix.sum(axis=0).A1
    avg_feature_scores = feature_sums / combined_tfidf_matrix.shape[0]
    sorted_indices = avg_feature_scores.argsort()[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]

    with open(output_feature_names_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_features, f, ensure_ascii=False, indent=4)

    print(f"Sorted feature names saved to {output_feature_names_path}")

if __name__ == "__main__":
    main()
