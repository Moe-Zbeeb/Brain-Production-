from fastapi import FastAPI
from routes.chatbotroutes import router  # Import the router from chatbot_routes.py

# Initialize FastAPI app
app = FastAPI()

#include a welcome message 
@app.get("/")
def read_root():
    return {"message": "Welcome to the Chatbot API!"}

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
