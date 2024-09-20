from fastapi import FastAPI,APIRouter 
import os 

base_router = APIRouter(
prefix = "/api/v1" ,
tags = ["Base"]   ,
)

@base_router.get("/") 
def read_root():    
    app_name = os.getenv("APP_NAME")  
    app_version = os.getenv("APP_VERSION")
    return {"message": "Hello api is working fine", "app_name": app_name, "app_version": app_version}  
