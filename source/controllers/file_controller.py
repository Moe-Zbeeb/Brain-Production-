# controllers/file_controller.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from helper.file_processing import process_file

file_router = APIRouter(
    prefix="/api/v1/files",
    tags=["File Upload"]
)

@file_router.post("/upload")
async def upload_file(file: UploadFile = File(...), index_name: str = None):
    try:
        # Validate the index name provided by the user
        if not index_name or not index_name.isalnum():
            raise HTTPException(status_code=400, detail="Invalid index name. It must be alphanumeric.")

        # Save the uploaded file to the assets folder
        file_location = f"assets/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # Determine file type (pdf, pptx, txt)
        file_extension = file.filename.split(".")[-1].lower()

        # Validate supported file types
        if file_extension not in ["pdf", "pptx", "txt"]:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        # Process the file and add to the specified index
        vectorstore = process_file(file_location, file_extension, index_name)

        return {"message": f"File '{file.filename}' processed and added to index '{index_name}' successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
