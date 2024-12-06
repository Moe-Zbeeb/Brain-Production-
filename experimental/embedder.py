# First, ensure you have the required libraries installed.
# You can install them using pip if you haven't already:

# !pip install sentence-transformers numpy

from sentence_transformers import SentenceTransformer
import numpy as np
import os

def embed_text_files(file_paths, model_name='all-MiniLM-L6-v2'):
    """
    Reads text from each file in file_paths, generates embeddings, and stores them in a dictionary.
    
    Args:
        file_paths (list of str): List of paths to .txt files.
        model_name (str): The name of the pre-trained SentenceTransformer model to use.
                          Defaults to 'all-MiniLM-L6-v2'.
    
    Returns:
        dict: A dictionary where keys are file names and values are embedding vectors (numpy arrays).
    """
    # Initialize the model
    model = SentenceTransformer(model_name)
    
    embeddings = {}
    
    for file_path in file_paths:
        try:
            # Ensure the file exists
            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}")
                continue
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            if not text.strip():
                print(f"File is empty: {file_path}")
                continue
            
            # Generate the embedding
            embedding = model.encode(text, convert_to_numpy=True)
            
            # Store the embedding with the file name as the key
            file_name = os.path.basename(file_path)
            embeddings[file_name] = embedding
            
            print(f"Embedded '{file_name}' successfully.")
        
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
    
    return embeddings

# Example usage:
if __name__ == "__main__":
    # List of text file paths
    txt_files = [
        'experimental/transcripts/Microsoft Teams Meetings ｜ How to Record,Trim or Cut, Share and Download ｜ Microsoft Streams & Video.mp3_transcript.txt',
    ]
    
    # Generate embeddings
    embeddings_dict = embed_text_files(txt_files)
    
    # Example: Accessing the embedding of 'document1.txt'
    if 'document1.txt' in embeddings_dict:
        print(f"Embedding for 'document1.txt':\n{embeddings_dict['document1.txt']}\n")
    
    # Optionally, convert the embeddings to a matrix for further processing
    # embedding_matrix = np.array(list(embeddings_dict.values()))
