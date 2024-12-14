from sklearn.feature_extraction.text import TfidfVectorizer
import json
import numpy as np
from scipy.sparse import save_npz, vstack

# Define file paths
cleaned_data_path = r"C:\Users\A1157\Downloads\Lab2new40123\fully_cleaned_data.json"
output_tfidf_matrix_path = r"C:\Users\A1157\Downloads\Lab2new40123\tfidf_matrix.npz"
output_feature_names_path = r"C:\Users\A1157\Downloads\Lab2new40123\tfidf_features_sorted.json"

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
    # Initialize variables
    vectorizer = None
    tfidf_matrices = []

    # Process each chunk and calculate TF-IDF
    for chunk in load_data_in_chunks(cleaned_data_path):
        tfidf_matrix, vectorizer = process_chunk_with_tfidf(chunk, vectorizer)
        tfidf_matrices.append(tfidf_matrix)
    
    # Combine all sparse matrices into a single matrix
    combined_tfidf_matrix = vstack(tfidf_matrices)

    # Save the TF-IDF matrix in sparse format
    save_npz(output_tfidf_matrix_path, combined_tfidf_matrix)
    print(f"TF-IDF matrix saved to {output_tfidf_matrix_path}")

    # Calculate the average TF-IDF value for each feature
    feature_names = vectorizer.get_feature_names_out()
    # Calculate the sum of TF-IDF scores for each feature (axis=0 for column-wise)
    feature_sums = combined_tfidf_matrix.sum(axis=0).A1
    # Calculate the average score by dividing by the number of documents
    avg_feature_scores = feature_sums / combined_tfidf_matrix.shape[0]

    # Sort features by their average TF-IDF score in descending order
    sorted_indices = avg_feature_scores.argsort()[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]

    # Save the sorted features to a JSON file
    with open(output_feature_names_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_features, f, ensure_ascii=False, indent=4)

    print(f"Sorted feature names saved to {output_feature_names_path}")

if __name__ == "__main__":
    main()
