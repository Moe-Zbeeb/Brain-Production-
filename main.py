from fastapi import FastAPI 
from routes.base import base 

app = FastAPI() 
app.include_router(base.base_router)  
