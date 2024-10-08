import os
from fastapi import FastAPI
from routes.transcription_routes import router as transcription_router  # Import transcription routes

# Initialize FastAPI app
app = FastAPI()

# Include a welcome message route
@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}

# Include the transcription routes
app.include_router(transcription_router, prefix="/video")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
