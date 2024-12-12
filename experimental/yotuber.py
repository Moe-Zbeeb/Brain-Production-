
import urllib.request
from bs4 import BeautifulSoup

textToSearch = 'hello world'
query = urllib.parse.quote(textToSearch)
url = "https://www.youtube.com/results?search_query=" + query
response = urllib.request.urlopen(url)
html = response.read()
soup = BeautifulSoup(html, 'html.parser')
for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
    print('https://www.youtube.com' + vid['href'])
    
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


def generate_youtube_title_with_langchain(api_key, query):
    """
    Generate a YouTube search title using LangChain with OpenAI's API.
    
    Parameters:
        api_key (str): Your OpenAI API key.
        query (str): The search query or keyword description.  
        
    Returns:
        str: A suggested YouTube video title.
    """
    # Initialize the OpenAI chat model via LangChain
    chat = ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Define the system and user prompts
    messages = [
        SystemMessage(content="You are an expert at generating YouTube keyword to search for."),
        HumanMessage(content=f"I dont know what to type on youtube generate for me thing (keyword) or topic to search for based on: {query} create the thing to search for."),
    ]
    
    try:
        # Use the LangChain chat model to get a response
        response = chat(messages)
        # Extract the content of the response
        return response.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Example Usage 

OPENAI_API_KEY = "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"  # Ensure this environment variable is set
query = "I need to know about animals. Keyword: Animal documentation"
title = generate_youtube_title_with_langchain(OPENAI_API_KEY, query)
print(f"Suggested YouTube Title: {title}")
    