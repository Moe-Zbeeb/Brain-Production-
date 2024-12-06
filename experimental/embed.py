import os
import assemblyai as aai
from pytube import YouTube

# Set up AssemblyAI API
aai.settings.api_key = "76e966abc56746f88f365735a37c766f"  # Replace with your API key

def process_youtube_links(youtube_links, output_dir="transcripts"):
    """
    Downloads audio from YouTube links, transcribes them, and saves transcripts with speaker labels.
    
    Parameters:
        youtube_links (list): List of YouTube video URLs.
        output_dir (str): Directory to save the transcripts.
        
    Returns:
        dict: A dictionary where keys are video titles, and values are the transcript text.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    transcriber = aai.Transcriber()
    transcripts = {}  # Store transcripts for further use

    for link in youtube_links:
        try:
            # Step 1: Download YouTube audio
            yt = YouTube(link)
            print(f"Downloading audio for: {yt.title}")
            audio_stream = yt.streams.filter(only_audio=True).first()
            audio_file = os.path.join(output_dir, f"{yt.title}.mp3")
            audio_stream.download(output_path=output_dir, filename=f"{yt.title}.mp3")
            
            # Step 2: Transcribe audio with speaker diarization
            print(f"Transcribing: {yt.title}")
            config = aai.TranscriptionConfig(speaker_labels=True)
            transcript = transcriber.transcribe(audio_file, config)
            
            # Step 3: Save transcript to a text file
            transcript_file = os.path.join(output_dir, f"{yt.title}_transcript.txt")
            with open(transcript_file, "w") as f:
                for utterance in transcript.utterances:
                    f.write(f"Speaker {utterance.speaker}: {utterance.text}\n")
            
            print(f"Transcript saved: {transcript_file}")
            
            # Store transcript in dictionary
            transcript_text = "\n".join(
                [f"Speaker {utterance.speaker}: {utterance.text}" for utterance in transcript.utterances]
            )
            transcripts[yt.title] = transcript_text

        except Exception as e:
            print(f"Error processing {link}: {e}")
    
    return transcripts

# Example Usage
youtube_links = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=example2"
]
transcripts = process_youtube_links(youtube_links)

# Print transcript summaries
for title, transcript in transcripts.items():
    print(f"Transcript for {title[:30]}...:\n{transcript[:100]}...\n")
