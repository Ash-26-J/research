import os
import sys
import string
from itertools import combinations

# Optional: install via pip install matplotlib seaborn
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False


def read_file(filepath):
    """Read and return content of a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().lower()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)


def tokenize(text):
    """Tokenize text by removing punctuation and splitting into unique words."""
    return set(text.translate(str.maketrans('', '', string.punctuation)).split())


def jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union != 0 else 0


def compute_similarity_matrix(token_sets):
    """Generate a matrix of Jaccard similarities between all pairs."""
    size = len(token_sets)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = jaccard_similarity(token_sets[i], token_sets[j])
    return matrix


def print_similarity_table(headers, matrix):
    """Print similarity matrix in tabular format."""
    print("\n📊 Jaccard Similarity Matrix:")
    print("\t" + "\t".join(headers))
    for i, row in enumerate(matrix):
        formatted_row = [f"{val:.4f}" for val in row]
        print(f"{headers[i]}\t" + "\t".join(formatted_row))


def plot_similarity_heatmap(headers, matrix):
    """Plot similarity matrix as a heatmap."""
    if not HAS_VISUALIZATION:
        print("⚠️ Visualization skipped: matplotlib/seaborn not installed.")
        return

    try:
        import numpy as np
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=headers, yticklabels=headers, cmap="YlGnBu")
        plt.title("LLM Output Similarity - Jaccard Index")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization error: {e}")


def main(file_paths):
    # Read files
    contents = [read_file(fp) for fp in file_paths]
    headers = [os.path.basename(fp) for fp in file_paths]

    # Tokenize content
    token_sets = [tokenize(text) for text in contents]

    # Compute similarity matrix
    matrix = compute_similarity_matrix(token_sets)

    # Display results
    print_similarity_table(headers, matrix)
    plot_similarity_heatmap(headers, matrix)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("1. python compare_llms.py file1.txt file2.txt ... file5.txt")
        print("2. Or drag and drop 5 text files onto this script.")
        sys.exit(1)

    file_paths = sys.argv[1:]

    # If folder is provided instead of files
    if len(file_paths) == 1 and os.path.isdir(file_paths[0]):
        folder_path = file_paths[0]
        file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Validate number of files
    if len(file_paths) != 5:
        print(f"❌ Expected 5 text files, but got {len(file_paths)}.")
        sys.exit(1)

    main(file_paths)
