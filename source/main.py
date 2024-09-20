# main.py

from fastapi import FastAPI
from routes import base
from controllers.file_controller import file_router
from controllers.qa_controller import qa_router

app = FastAPI()

# Include routers
app.include_router(base.base_router)
app.include_router(file_router)
app.include_router(qa_router)

