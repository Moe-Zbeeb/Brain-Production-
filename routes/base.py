from fastapi import FastAPI,APIRouter 

base_router = APIRouter() 

@base_router.get("/") 
def read_root(): 
    return {"message": "Hello api is working"}  
