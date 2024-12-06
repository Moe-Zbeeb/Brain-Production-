import os
import re
import subprocess
import assemblyai as aai

aai.settings.api_key = "76e966abc56746f88f365735a37c766f"  # Replace with your API key

def validate_youtube_url(url):
    """Validate if a URL is a valid YouTube video link."""
    pattern = r"^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$"
    return re.match(pattern, url) is not None

def download_audio_yt_dlp(video_url, output_dir):
    """Download audio using yt-dlp with a custom user agent."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "%(title)s.%(ext)s")
        command = [
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "--output", output_file,
            video_url
        ]
        subprocess.run(command, check=True)
        print(f"Downloaded audio for: {video_url}")
        return True
    except Exception as e:
        print(f"Error downloading audio for {video_url}: {e}")
        return False

def process_youtube_links(youtube_links, output_dir="transcripts"):
    """Download audio from YouTube links, transcribe them, and save transcripts."""
    os.makedirs(output_dir, exist_ok=True)
    transcriber = aai.Transcriber()
    transcripts = {}

    for link in youtube_links:
        if not validate_youtube_url(link):
            print(f"Invalid YouTube URL: {link}")
            continue

        print(f"Processing: {link}")
        if not download_audio_yt_dlp(link, output_dir):
            continue
        
        try:
            # Find the downloaded MP3 file
            audio_file = next(
                (f for f in os.listdir(output_dir) if f.endswith(".mp3")), None
            )
            if audio_file:
                audio_path = os.path.join(output_dir, audio_file)
                config = aai.TranscriptionConfig(speaker_labels=True)
                transcript = transcriber.transcribe(audio_path, config)
                
                # Save transcript to a text file
                transcript_file = os.path.join(output_dir, f"{audio_file}_transcript.txt")
                with open(transcript_file, "w") as f:
                    for utterance in transcript.utterances:
                        f.write(f"Speaker {utterance.speaker}: {utterance.text}\n")
                
                print(f"Transcript saved: {transcript_file}")
                
                # Store transcript in dictionary
                transcript_text = "\n".join(
                    [f"Speaker {utterance.speaker}: {utterance.text}" for utterance in transcript.utterances]
                )
                transcripts[audio_file] = transcript_text
        except Exception as e:
            print(f"Error processing {link}: {e}")
    
    return transcripts

# Example Usage
youtube_links = [
    "https://www.youtube.com/watch?v=NIvwRKgU9Sk",  # Replace with valid links
    "https://www.youtube.com/watch?v=iqBKZuWnogM"       # Replace with valid links
]

transcripts = process_youtube_links(youtube_links)
