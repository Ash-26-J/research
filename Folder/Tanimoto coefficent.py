import os
import sys

def read_file(filepath):
    """Read the content of a file and return lowercase text."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().lower()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)

def tokenize(text):
    """Tokenize text into unique words (set)."""
    import string
    # Remove punctuation and split into words
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return set(text.split())

def tanimoto_coefficient(set_a, set_b):
    """Compute Tanimoto Coefficient (Jaccard Index) between two sets."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union != 0 else 0.0

def build_similarity_matrix(sets):
    """Build Tanimoto similarity matrix for list of sets."""
    size = len(sets)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = tanimoto_coefficient(sets[i], sets[j])
    return matrix

def print_similarity_table(headers, matrix):
    """Print similarity matrix in tabular format."""
    print("\n🧬 Tanimoto Coefficient Matrix:")
    print("\t" + "\t".join(headers))
    for i, row in enumerate(matrix):
        formatted_row = [f"{val:.4f}" for val in row]
        print(f"{headers[i]}\t" + "\t".join(formatted_row))

def main(file_paths):
    # Step 1: Read and tokenize each file
    contents = [read_file(fp) for fp in file_paths]
    token_sets = [tokenize(text) for text in contents]

    # Step 2: Build similarity matrix
    matrix = build_similarity_matrix(token_sets)

    # Step 3: Print results
    headers = [os.path.basename(fp) for fp in file_paths]
    print_similarity_table(headers, matrix)

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
