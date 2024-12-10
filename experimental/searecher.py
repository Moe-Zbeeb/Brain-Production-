import urllib.request
from bs4 import BeautifulSoup
import re

def search_youtube(keyword, num_results=10):
    """
    Search YouTube and return the specified number of video links.
    
    Parameters:
        keyword (str): The search keyword.
        num_results (int): The number of video links to return.
        
    Returns:
        list: A list of YouTube video links.
    """
    search_keyword = keyword.replace(" ", "+")
    url = f"https://www.youtube.com/results?search_query={search_keyword}"
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract video IDs using regex
    video_ids = re.findall(r"watch\?v=(\S{11})", str(soup))
    video_links = [f"https://www.youtube.com/watch?v={video_id}" for video_id in video_ids[:num_results]]
    
    if video_links:
        return video_links
    else:
        return ["No videos found."]

# Example usage
links = search_youtube("mozart", 10)
for idx, link in enumerate(links, 1):
    print(f"Video {idx}: {link}")
