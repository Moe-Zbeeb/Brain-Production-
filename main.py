import os
from fastapi import FastAPI
from routes.chatbotroutes import router  # Ensure this import is correct
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Welcome route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Chatbot API!"}

# Include the router for the chatbot functionality
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    # Run the application using Uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
