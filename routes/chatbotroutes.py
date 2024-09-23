from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from chatbot.chatbot import ChatBot  # Import the ChatBot class from chatbot.py

# Create a router object
router = APIRouter()

# Dictionary to store ChatBot instances by index name
bots = {}


class TrainRequest(BaseModel):
    index_name: str
    file_paths: List[str]


class AskRequest(BaseModel):
    index_name: str
    question: str


@router.post("/initialize")
def initialize_bot(index_name: str):
    """
    Initialize a ChatBot instance with the given index name.

    Args:
        index_name (str): The name of the Pinecone index to initialize.

    Returns:
        A success message if initialization is successful.
    """
    try:
        # Create a new ChatBot instance and store it in the dictionary
        bot = ChatBot(index_name)
        bots[index_name] = bot
        return {"message": f"ChatBot for index '{index_name}' initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
def train_bot(request: TrainRequest):
    bot = bots.get(request.index_name)
    if not bot:
        raise HTTPException(status_code=404, detail="ChatBot instance not found.")
    
    try:
        bot.train(request.file_paths)
        return {"message": f"ChatBot for index '{request.index_name}' trained successfully."}
    except Exception as e:
        # Log the detailed exception for debugging purposes
        print(f"Error during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/ask")
def ask_question(request: AskRequest):
    """
    Ask a question to the ChatBot instance and retrieve an answer.

    Args:
        request (AskRequest): The request body containing index_name and question.

    Returns:
        The answer from the ChatBot.
    """
    bot = bots.get(request.index_name)
    if not bot:
        raise HTTPException(status_code=404, detail="ChatBot instance not found.")
    
    try:
        # Ask the question and return the answer
        answer = bot.ask_question(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
