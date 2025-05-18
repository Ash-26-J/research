import os
import sys
from collections import Counter

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

def build_binary_vectors(tokenized_docs, vocab):
    """Convert token lists into binary presence/absence vectors over the vocabulary."""
    vectors = []
    for tokens in tokenized_docs:
        word_set = set(tokens)
        vector = [1 if word in word_set else 0 for word in vocab]
        vectors.append(vector)
    return vectors

def simple_matching_coefficient(vec_a, vec_b):
    """Compute Simple Matching Coefficient between two binary vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of equal length.")
    
    matches = sum(a == b for a, b in zip(vec_a, vec_b))
    total = len(vec_a)
    return matches / total if total > 0 else 0.0

def build_similarity_matrix(vectors):
    """Build similarity matrix using SMC for all document pairs."""
    size = len(vectors)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = simple_matching_coefficient(vectors[i], vectors[j])
    return matrix

def main(file_paths):
    # Step 1: Read and tokenize all documents
    raw_texts = [read_file(fp) for fp in file_paths]
    tokenized_docs = [tokenize(text) for text in raw_texts]

    # Step 2: Build global vocabulary
    vocab = sorted(set(word for tokens in tokenized_docs for word in tokens))

    # Step 3: Create binary vectors
    binary_vectors = build_binary_vectors(tokenized_docs, vocab)

    # Step 4: Compute similarity matrix
    matrix = build_similarity_matrix(binary_vectors)

    # Step 5: Print similarity table
    headers = [os.path.basename(fp) for fp in file_paths]
    print("\n📊 Simple Matching Coefficient Similarity Matrix:")
    print("\t" + "\t".join(headers))
    for i in range(len(matrix)):
        row = [headers[i]] + [f"{val:.4f}" for val in matrix[i]]
        print("\t".join(row))

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python compare_llms_smc.py file1.txt file2.txt file3.txt file4.txt file5.txt")
        sys.exit(1)

    file_paths = sys.argv[1:]
    for fp in file_paths:
        if not os.path.isfile(fp):
            print(f"File not found: {fp}")
            sys.exit(1)

    main(file_paths)
