from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
import urllib.request
from bs4 import BeautifulSoup
import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def generate_youtube_title_with_langchain(api_key, query):
    """
    Generate a YouTube search title using LangChain with OpenAI's API.
    """
    chat = ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    messages = [
        SystemMessage(content="You are an expert at generating YouTube keyword to search for."),
        HumanMessage(content=f"I need help finding a good keyword to search for on YouTube based on this topic: {query}")
    ]
    try:
        response = chat(messages)
        return response.content.strip()
    except Exception as e:
        return f"Error: {e}"

def search_youtube(keyword, num_results=10):
    """
    Search YouTube and return the specified number of video links.
    """
    search_keyword = keyword.replace(" ", "+")
    url = f"https://www.youtube.com/results?search_query={search_keyword}"
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')

    video_ids = re.findall(r"watch\?v=(\S{11})", str(soup))
    video_links = [f"https://www.youtube.com/watch?v={video_id}" for video_id in video_ids[:num_results]]

    return video_links if video_links else ["No videos found."]

def embed_text_files(file_paths, model_name='all-MiniLM-L6-v2'):
    """
    Reads text from each file in file_paths, generates embeddings, and stores them in a dictionary.
    """
    model = SentenceTransformer(model_name)
    embeddings = {}

    for file_path in file_paths:
        try:
            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}")
                continue

            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            if not text.strip():
                print(f"File is empty: {file_path}")
                continue

            embedding = model.encode(text, convert_to_numpy=True)
            file_name = os.path.basename(file_path)
            embeddings[file_name] = embedding

            print(f"Embedded '{file_name}' successfully.")

        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    return embeddings

def recommend_videos(query_embedding, video_embeddings, top_n=2):
    """
    Recommend top N videos based on similarity to the query embedding.
    """
    video_names = list(video_embeddings.keys())
    embeddings = np.array(list(video_embeddings.values()))
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    top_indices = similarities.argsort()[-top_n:][::-1]

    recommendations = [(video_names[i], similarities[i]) for i in top_indices]
    return recommendations

def main():
    OPENAI_API_KEY = "your-openai-api-key"  # Replace with your actual OpenAI API key
    query = input("Enter your query: ")

    # Step 1: Generate a refined search query
    refined_query = generate_youtube_title_with_langchain(OPENAI_API_KEY, query)
    print(f"Refined Query: {refined_query}")

    # Step 2: Search YouTube
    video_links = search_youtube(refined_query, num_results=5)
    print(f"Video Links: {video_links}")

    # Simulating transcript files (replace with actual paths after downloading transcripts)
    transcript_files = [
        "video1_transcript.txt",  # Replace with actual paths
        "video2_transcript.txt",
    ]

    # Step 3: Embed transcripts
    video_embeddings = embed_text_files(transcript_files)

    # Step 4: Compare embeddings and recommend videos
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_numpy=True)
    recommendations = recommend_videos(query_embedding, video_embeddings)

    print("Top Recommended Videos:")
    for rec in recommendations:
        print(f"Video: {rec[0]}, Similarity: {rec[1]}")

if __name__ == "__main__":
    main()
