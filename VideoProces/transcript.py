import os
import whisper
from moviepy.editor import VideoFileClip

class VideoTranscriptionService:
    def __init__(self):
        self.upload_dir = "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)

    # Function to extract audio from a video file
    def extract_audio(self, video_path, output_audio_path="audio.wav"):
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path)
        print(f"Audio extracted and saved to {output_audio_path}")
        return output_audio_path

    # Function to transcribe audio using Whisper
    def transcribe_audio_whisper(self, audio_path, model_size="base"):
        # Load the Whisper model
        model = whisper.load_model(model_size)
        print(f"Using Whisper model: {model_size}")

        # Transcribe the audio
        result = model.transcribe(audio_path)

        # Return the transcription result
        return result

    # Function to save transcription to a file
    def save_transcription_to_file(self, transcription, output_file="transcription.txt"):
        with open(output_file, 'w') as f:
            for segment in transcription['segments']:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text']
                f.write(f"[{start_time:.2f} - {end_time:.2f}] {text}\n")
        print(f"Transcription saved to {output_file}")

    # Function to handle full video processing
    def process_video(self, video_path, model_size="base"):
        # Extract audio from the video
        audio_path = self.extract_audio(video_path)
        
        # Transcribe the extracted audio using Whisper
        transcription = self.transcribe_audio_whisper(audio_path, model_size=model_size)
        
        # Save the transcription to a text file
        self.save_transcription_to_file(transcription, output_file="transcription.txt")
        
        return transcription