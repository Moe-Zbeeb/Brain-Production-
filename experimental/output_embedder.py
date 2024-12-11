OPENAI_API_KEY = "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"  # Ensure this environment variable is set
from langchain_openai import ChatOpenAI  # Updated import
from langchain.schema import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
import urllib.request
from bs4 import BeautifulSoup
import re
import os
import chardet
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI API Key (replace with your actual key)

def generate_youtube_keyword(api_key, query):
    """
    Generate a YouTube search keyword using LangChain.
    """
    chat = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0.7)
    messages = [
        SystemMessage(content="You are an expert at generating YouTube search keywords."),
        HumanMessage(content=f"Suggest a good YouTube search keyword for this topic: {query}")
    ]
    try:
        response = chat.invoke(messages)
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
    return [f"https://www.youtube.com/watch?v={video_id}" for video_id in video_ids[:num_results]] if video_ids else []

def download_transcripts(video_links, folder_path="transcripts"):
    """
    Download video transcripts and save them to a folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    transcripts = {}
    for i, video_link in enumerate(video_links):
        video_id = video_link.split("=")[-1]
        transcript_path = os.path.join(folder_path, f"transcript_{video_id}.txt")
        # Simulated transcript fetching (replace with actual API if needed)
        fake_transcript = f"Transcript for video {video_id}"  # Replace with actual transcript fetching logic
        with open(transcript_path, 'w', encoding='utf-8') as file:
            file.write(fake_transcript)
        transcripts[video_id] = fake_transcript

    return transcripts

def embed_transcripts(transcripts, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings for transcripts.
    """
    model = SentenceTransformer(model_name)
    embeddings = {}
    for video_id, transcript in transcripts.items():
        embedding = model.encode(transcript, convert_to_numpy=True)
        embeddings[video_id] = embedding
    return embeddings

def recommend_video(query_embedding, video_embeddings):
    """
    Recommend the most relevant video based on similarity to the query embedding.
    """
    video_ids = list(video_embeddings.keys())
    embeddings = np.array(list(video_embeddings.values()))
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_index = similarities.argmax()
    return video_ids[top_index], similarities[top_index]

def main():
    query = input("Enter your query: ").strip()
    if not query:
        print("Query cannot be empty.")
        return

    # Step 1: Generate a refined YouTube search keyword
    refined_query = generate_youtube_keyword(OPENAI_API_KEY, query)
    print(f"Generated Keyword: {refined_query}")

    # Step 2: Search YouTube for videos
    video_links = search_youtube(refined_query, num_results=3)
    if not video_links:
        print("No videos found. Please refine your query.")
        return
    print(f"Video Links: {video_links}")

    # Step 3: Download transcripts for the top 3 videos
    transcripts = download_transcripts(video_links)

    # Step 4: Embed transcripts
    video_embeddings = embed_transcripts(transcripts)

    # Step 5: Generate query embedding and recommend the best video
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_numpy=True)
    best_video_id, similarity = recommend_video(query_embedding, video_embeddings)

    print(f"Best Video: https://www.youtube.com/watch?v={best_video_id}, Similarity: {similarity}")

if __name__ == "__main__":
    main()
