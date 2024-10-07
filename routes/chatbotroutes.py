from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from chatbot.chatbot import ChatBot  # Import the ChatBot class from the correct file

app = FastAPI()

# Create a router object
router = APIRouter()

# Dictionary to store ChatBot instances by index name
bots = {}


# Pydantic model for requests
class TrainRequest(BaseModel):
    index_name: str
    file_paths: List[str]


class AskRequest(BaseModel):
    index_name: str
    question: str


# Route to initialize the ChatBot instance
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


# Route to train the ChatBot instance
@router.post("/train")
def train_bot(request: TrainRequest):
    """
    Train the ChatBot instance with PDF files.
    
    Args:
        request (TrainRequest): The request body containing index name and file paths.

    Returns:
        Success message if training is successful.
    """
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
        answer = bot.ask_question(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Route to reset the ChatBot instance
@router.post("/reset")
def reset_bot(index_name: str):
    """
    Reset the Pinecone index for a specific ChatBot instance.

    Args:
        index_name (str): The name of the index to reset.

    Returns:
        A success message if the reset is successful.
    """
    bot = bots.get(index_name)
    if not bot:
        raise HTTPException(status_code=404, detail="ChatBot instance not found.")
    
    try:
        bot.reset()
        return {"message": f"ChatBot for index '{index_name}' has been reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# Include the router in the FastAPI app
