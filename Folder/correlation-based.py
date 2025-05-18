import os
import sys
from collections import Counter
import math

def read_file(filepath):
    """Read content of a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().lower()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)

def tokenize(text):
    """Remove punctuation and split into words."""  
    punct = set("!\"#$%&'()*+,-./:;<=>?@[\$$^_`{|}~")
    return [word for word in ''.join(c if c not in punct else ' ' for c in text).split()]

def compute_frequency_vector(tokens, vocab):
    """Create frequency vector based on vocabulary."""
    freq = Counter(tokens)
    return [freq.get(word, 0) for word in vocab]

def pearson_correlation(vec_a, vec_b):
    """Compute Pearson correlation coefficient between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of equal length.")
    
    n = len(vec_a)
    if n == 0:
        return 0.0

    sum_x = sum(vec_a)
    sum_y = sum(vec_b)
    mean_x = sum_x / n
    mean_y = sum_y / n

    numerator = sum((vec_a[i] - mean_x) * (vec_b[i] - mean_y) for i in range(n))
    denom_x = math.sqrt(sum((x - mean_x)**2 for x in vec_a))
    denom_y = math.sqrt(sum((y - mean_y)**2 for y in vec_b))

    if denom_x == 0 or denom_y == 0:
        return 0.0  # No variation in one or both vectors

    return numerator / (denom_x * denom_y)

def build_similarity_matrix(vectors):
    """Build Pearson correlation matrix between all document vectors."""
    size = len(vectors)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = pearson_correlation(vectors[i], vectors[j])
    return matrix

def main(file_paths):
    # Step 1: Read and tokenize all documents
    raw_texts = [read_file(fp) for fp in file_paths]
    tokenized_docs = [tokenize(text) for text in raw_texts]

    # Step 2: Build global vocabulary
    vocab = sorted(set(word for tokens in tokenized_docs for word in tokens))

    # Step 3: Create frequency vectors
    vectors = [compute_frequency_vector(tokens, vocab) for tokens in tokenized_docs]

    # Step 4: Compute Pearson correlation matrix
    matrix = build_similarity_matrix(vectors)

    # Step 5: Print similarity table
    headers = [os.path.basename(fp) for fp in file_paths]
    print("\n📊 Correlation-Based Similarity Matrix (Pearson Correlation):")
    print("\t" + "\t".join(headers))
    for i in range(len(matrix)):
        row = [headers[i]] + [f"{val:.4f}" for val in matrix[i]]
        print("\t".join(row))

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python compare_llms.py file1.txt file2.txt file3.txt file4.txt file5.txt")
        sys.exit(1)

    file_paths = sys.argv[1:]
    for fp in file_paths:
        if not os.path.isfile(fp):
            print(f"File not found: {fp}")
            sys.exit(1)

    main(file_paths)
