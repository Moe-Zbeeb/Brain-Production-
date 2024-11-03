from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from chatbot.chatbot import ChatBot  # Import the ChatBot class from the correct file

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a router object
router = APIRouter()

# Dictionary to store ChatBot instances by index name
bots = {}

# Pydantic model for requests
class InitializeRequest(BaseModel):
    index_name: str

class TrainRequest(BaseModel):
    index_name: str
    file_paths: List[str]

class AskRequest(BaseModel):
    index_name: str
    question: str

# Route to initialize the ChatBot instance
@router.post("/initialize")
def initialize_bot(request: InitializeRequest):
    try:
        bot = ChatBot(request.index_name)
        bots[request.index_name] = bot
        return {"message": f"ChatBot for index '{request.index_name}' initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to train the ChatBot instance
@router.post("/train")
def train_bot(request: TrainRequest):
    bot = bots.get(request.index_name)
    if not bot:
        raise HTTPException(status_code=404, detail="ChatBot instance not found.")
    try:
        bot.train(request.file_paths)
        return {"message": f"ChatBot for index '{request.index_name}' trained successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# Route to ask a question to the ChatBot
@router.post("/ask")
def ask_question(request: AskRequest):
    bot = bots.get(request.index_name)
    if not bot:
        raise HTTPException(status_code=404, detail="ChatBot instance not found.")
    try:
        answer = bot.ask_question(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to reset the ChatBot instance
@router.post("/reset")
def reset_bot(index_name: str):
    bot = bots.get(index_name)
    if not bot:
        raise HTTPException(status_code=404, detail="ChatBot instance not found.")
    try:
        bot.reset()
        return {"message": f"ChatBot for index '{index_name}' has been reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

# Include the router in the FastAPI app
app.include_router(router, prefix="/api/v1")
