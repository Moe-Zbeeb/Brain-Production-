import os
from fastapi import FastAPI
from routes.transcription_routes import router as transcription_router  # Routes for video transcription
from routes.chatbotroutes import router as chatbot_router  # Routes for chatbot

# Load environment variables
os.environ['PINECONE_API_KEY'] = 'your_pinecone_api_key'
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'

# Initialize FastAPI app
app = FastAPI()

# Include a welcome message route
@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}

# Include the transcription routes
app.include_router(transcription_router, prefix="/video")

# Include the chatbot routes
app.include_router(chatbot_router, prefix="/chatbot")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
