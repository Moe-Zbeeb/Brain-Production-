import os
import pickle
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def embed_text_files(texts_dir, model_name='all-MiniLM-L6-v2'):
    """
    Embeds all .txt files in the specified directory using a SentenceTransformer model.

    Args:
        texts_dir (str): Path to the directory containing .txt files.
        model_name (str): Name of the pre-trained SentenceTransformer model.

    Returns:
        dict: A dictionary mapping file names to their embedding vectors.
    """
    model = SentenceTransformer(model_name)
    embeddings = {}

    # List all .txt files in the directory
    try:
        txt_files = [f for f in os.listdir(texts_dir) if f.endswith('.txt')]
    except FileNotFoundError:
        logger.error(f"Directory '{texts_dir}' does not exist.")
        return embeddings

    if not txt_files:
        logger.warning(f"No .txt files found in directory '{texts_dir}'.")
        return embeddings

    logger.info(f"Embedding {len(txt_files)} files from '{texts_dir}' using model '{model_name}'...")

    for file_name in tqdm(txt_files, desc="Embedding files"):
        file_path = os.path.join(texts_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
                if not text:
                    logger.warning(f"Warning: '{file_name}' is empty. Skipping.")
                    continue
                embedding = model.encode(text, convert_to_numpy=True).astype('float32')
                embeddings[file_name] = embedding
        except Exception as e:
            logger.error(f"Error processing '{file_name}': {e}")

    logger.info(f"Completed embedding {len(embeddings)} files.")
    return embeddings


def save_embeddings(embeddings, file_path):
    """Save embeddings to a pickle file."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"Embeddings saved to '{file_path}'.")
    except Exception as e:
        logger.error(f"Failed to save embeddings to '{file_path}': {e}")


def load_embeddings(file_path):
    """Load embeddings from a pickle file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Embeddings file '{file_path}' not found.")
    try:
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"Embeddings loaded from '{file_path}'.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings from '{file_path}': {e}")
        return {}


def build_faiss_index(embeddings, metric='cosine'):
    """
    Builds a FAISS index from the provided embeddings.

    Args:
        embeddings (dict): Dictionary mapping file names to embedding vectors.
        metric (str): Distance metric ('cosine' is effectively handled via inner product with normalized vectors).

    Returns:
        faiss.Index: Built FAISS index.
    """
    file_names = list(embeddings.keys())
    if not file_names:
        logger.error("No embeddings available to build FAISS index.")
        return None

    embedding_vectors = np.array([embeddings[file] for file in file_names]).astype('float32')

    faiss.normalize_L2(embedding_vectors)  # Normalize for cosine similarity
    dim = embedding_vectors.shape[1]

    if metric == 'cosine':
        index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity
        logger.info("Using Inner Product (IP) for cosine similarity.")
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Only 'cosine' is supported.")

    index.add(embedding_vectors)
    logger.info(f"FAISS index built with {index.ntotal} vectors.")
    return index


def save_faiss_index(index, file_path):
    """Save FAISS index to a file."""
    try:
        faiss.write_index(index, file_path)
        logger.info(f"FAISS index saved to '{file_path}'.")
    except Exception as e:
        logger.error(f"Failed to save FAISS index to '{file_path}': {e}")


def load_faiss_index(file_path):
    """Load FAISS index from a file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"FAISS index file '{file_path}' not found.")
    try:
        index = faiss.read_index(file_path)
        logger.info(f"FAISS index loaded from '{file_path}'.")
        return index
    except Exception as e:
        logger.error(f"Failed to load FAISS index from '{file_path}': {e}")
        return None


def embed_query(query, model):
    """Embed the input query using the provided SentenceTransformer model."""
    try:
        query_embedding = model.encode(query, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        return query_embedding.reshape(1, -1)
    except Exception as e:
        logger.error(f"Failed to embed query '{query}': {e}")
        return None


def find_most_relevant_file(query_embedding, index, file_names, top_k=1):
    """Find the most relevant file(s) for a given query."""
    if query_embedding is None:
        logger.error("Query embedding is None. Cannot perform search.")
        return []

    similarity_scores, indices = index.search(query_embedding, top_k)

    # Log shapes and contents for debugging
    logger.debug(f"Similarity Scores Shape: {similarity_scores.shape}")
    logger.debug(f"Indices Shape: {indices.shape}")
    logger.debug(f"Similarity Scores: {similarity_scores}")
    logger.debug(f"Indices: {indices}")

    # Check if similarity_scores and indices have expected shapes
    if similarity_scores.shape != (1, top_k) or indices.shape != (1, top_k):
        logger.error(f"Unexpected shapes for similarity_scores {similarity_scores.shape} and indices {indices.shape}.")
        return []

    results = []
    for score, idx in zip(similarity_scores[0], indices[0]):
        if 0 <= idx < len(file_names):
            results.append((file_names[idx], score))
        else:
            logger.warning(f"Index {idx} out of bounds for file_names list of size {len(file_names)}.")

    return results


def main():
    parser = argparse.ArgumentParser(description="FAISS-based text search system.")
    parser.add_argument('--texts_dir', default='/home/mohammad/Brain-Production-/experimental/transcripts',
                        help='Directory containing .txt files.')
    parser.add_argument('--embeddings_file', default='embeddings.pkl',
                        help='Path to embeddings file.')
    parser.add_argument('--faiss_index_file', default='faiss_index.bin',
                        help='Path to FAISS index file.')
    parser.add_argument('--model_name', default='all-MiniLM-L6-v2',
                        help='SentenceTransformer model name.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of top results to retrieve.')
    args = parser.parse_args()

    try:
        # Load or create embeddings
        if not os.path.isfile(args.embeddings_file):
            logger.info("Embedding text files...")
            embeddings = embed_text_files(args.texts_dir, args.model_name)
            if not embeddings:
                logger.error("No embeddings generated. Exiting.")
                return
            save_embeddings(embeddings, args.embeddings_file)
        else:
            embeddings = load_embeddings(args.embeddings_file)

        # Load or create FAISS index
        if not os.path.isfile(args.faiss_index_file):
            logger.info("Building FAISS index...")
            index = build_faiss_index(embeddings)
            if index is None:
                logger.error("FAISS index could not be built. Exiting.")
                return
            save_faiss_index(index, args.faiss_index_file)
        else:
            index = load_faiss_index(args.faiss_index_file)
            if index is None:
                logger.error("FAISS index could not be loaded. Exiting.")
                return

        # Initialize the model
        logger.info(f"Loading SentenceTransformer model '{args.model_name}'...")
        model = SentenceTransformer(args.model_name)
        logger.info("Embedding model loaded successfully.")

        # Extract file names list for mapping indices to file names
        file_names = list(embeddings.keys())

        # Query Loop
        logger.info("=== FAISS k-NN Retrieval System ===")
        logger.info("Type 'exit' or 'quit' to terminate.\n")

        while True:
            query = input("Enter your query: ").strip()
            if query.lower() in ['exit', 'quit']:
                logger.info("Exiting...")
                break

            if not query:
                logger.warning("Empty query. Please try again.")
                continue

            # Embed the query
            query_embedding = embed_query(query, model)

            # Find the most relevant file(s)
            results = find_most_relevant_file(query_embedding, index, file_names, top_k=args.top_k)

            if results:
                for rank, (file, score) in enumerate(results, start=1):
                    print(f"Rank {rank}: File={file}, Score={round(score, 4)}")
            else:
                logger.warning("No relevant files found.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
