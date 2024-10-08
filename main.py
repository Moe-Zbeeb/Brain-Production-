import os
from fastapi import FastAPI
from routes.chatbotroutes import router  # Assuming you have this router for chatbot routes
import uvicorn

# Load environment variables for Pinecone and OpenAI API keys
os.environ['PINECONE_API_KEY'] = 'cd4f53f7-4807-4868-81dd-c15760aa0b36'
os.environ['OPENAI_API_KEY'] = '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'

# Initialize FastAPI app
app = FastAPI()

# Include a welcome message route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Chatbot API!"}

# Include the router for the chatbot functionality
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    # Run the application using Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
