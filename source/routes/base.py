from fastapi import FastAPI,APIRouter 
import os    
from helper.config import get_settings
base_router = APIRouter(
prefix = "/api/v1" ,
tags = ["Base"]   ,
)

@base_router.get("/") 
def read_root():      
    appsettings = get_settings()
    app_name = appsettings.APP_NAME
    app_version = appsettings.APP_VERSION
    return {"message": "Hello api is working fine", "app_name": app_name, "app_version": app_version}  
