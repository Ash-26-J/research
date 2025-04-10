import math
import os
from collections import Counter

def read_file(filepath):
    """Reads the content of a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().lower()  # Convert to lowercase for case-insensitive comparison
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def tokenize(text):
    """Simple tokenization: splits text into words (basic punctuation removal)."""
    return ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text).split()

def calculate_term_frequency(tokens):
    """Calculates the frequency of each term in a list of tokens."""
    return Counter(tokens)

def calculate_idf(term, all_documents):
    """Calculates the Inverse Document Frequency of a term across all documents."""
    num_documents_containing_term = sum(1 for doc in all_documents if term in doc)
    if num_documents_containing_term > 0:
        return math.log(len(all_documents) / num_documents_containing_term)
    else:
        return 0

def calculate_tfidf(term_frequency, all_documents):
    """Calculates the TF-IDF vector for a document."""
    tfidf_vector = {}
    for term, frequency in term_frequency.items():
        idf = calculate_idf(term, all_documents)
        tfidf_vector[term] = frequency * idf
    return tfidf_vector

def dot_product(vec1, vec2):
    """Calculates the dot product of two vectors (represented as dictionaries)."""
    common_terms = set(vec1.keys()) & set(vec2.keys())
    return sum(vec1.get(term, 0) * vec2.get(term, 0) for term in common_terms)

def magnitude(vec):
    """Calculates the magnitude (Euclidean norm) of a vector."""
    return math.sqrt(sum(val**2 for val in vec.values()))

def cosine_similarity(tfidf1, tfidf2):
    """Calculates the cosine similarity between two TF-IDF vectors."""
    dp = dot_product(tfidf1, tfidf2)
    mag1 = magnitude(tfidf1)
    mag2 = magnitude(tfidf1)
    if mag1 > 0 and mag2 > 0:
        return dp / (mag1 * mag2)
    else:
        return 0

def main():
    """Main function to get file inputs and display similarity for 5 files."""
    num_files_to_process = 5
    filenames = []
    for i in range(1, num_files_to_process + 1):
        while True:
            filepath = input(f"Enter the path for file {i}: ").strip()
            if os.path.exists(filepath):
                filenames.append(filepath)
                break
            else:
                print("Invalid file path. Please try again.")

    file_contents = [read_file(f) for f in filenames]
    if not all(file_contents):
        print("Error: One or more files could not be read.")
        return

    all_tokens = [tokenize(content) for content in file_contents]
    term_frequencies = [calculate_term_frequency(tokens) for tokens in all_tokens]
    all_documents_for_idf = [set(tokens) for tokens in all_tokens] # For IDF calculation

    tfidf_vectors = [calculate_tfidf(tf, all_documents_for_idf) for tf in term_frequencies]

    print("\nSimilarity Matrix:")
    print("-----------------" + "-" * (15 * num_files_to_process))
    print("        ", end="")
    for i in range(num_files_to_process):
        print(f"{os.path.basename(filenames[i]):<15}", end="")
    print()
    for i in range(num_files_to_process):
        print(f"{os.path.basename(filenames[i]):<8}", end="")
        for j in range(num_files_to_process):
            similarity = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
            print(f"{similarity:<15.4f}", end="")
        print()

    print("\nInterpretation of Similarity Scores (Cosine Similarity):")
    print("- 1.0: Identical content")
    print("- Closer to 1.0: High similarity")
    print("- Closer to 0.0: Low similarity (little to no overlap in important terms)")

if __name__ == "__main__":
    main()
