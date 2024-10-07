from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
from VideoProces.transcript import VideoTranscriptionService

# Initialize the transcription service
transcription_service = VideoTranscriptionService()

# Initialize FastAPI router
router = APIRouter()

# Route to upload a video file
@router.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded video file
        video_path = os.path.join(transcription_service.upload_dir, file.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Video file {file.filename} uploaded successfully.")
        return {"message": f"Video {file.filename} uploaded successfully.", "video_path": video_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video upload failed: {str(e)}")

# Route to extract audio and transcribe the video
@router.post("/transcribe/")
async def transcribe_video(video_path: str, model_size: str = "base"):
    try:
        # Process the video: extract audio, transcribe, and return the transcription
        transcription = transcription_service.process_video(video_path, model_size=model_size)
        return transcription
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
