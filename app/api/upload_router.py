from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pydantic import BaseModel
from urllib.parse import urlparse  # <--- needed
from app.db.s3_client import upload_file_to_s3, delete_file_from_s3  # <--- add delete_file_from_s3

router = APIRouter(tags=["Uploads"])


class DeleteImagesRequest(BaseModel):
    urls: List[str]  # Array of S3 URLs to delete


@router.post("/upload-images/")
async def upload_images(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    uploaded_urls = []

    for file in files:
        # Validate file type
        if file.content_type.split('/')[0] != "image":
            continue  # skip invalid files

        try:
            # Read file bytes
            file_bytes = await file.read()
            file_name = f"images/{file.filename}"  # store in 'images/' folder on S3

            # Upload to S3
            s3_url = upload_file_to_s3(file_bytes, file_name, file.content_type)
            if s3_url:
                uploaded_urls.append(s3_url)

        except Exception as e:
            print(f"❌ Failed to upload {file.filename}: {e}")
            continue

    if not uploaded_urls:
        raise HTTPException(status_code=500, detail="No valid images were uploaded")

    return {"uploaded_images": uploaded_urls}


@router.delete("/delete-images/")
async def delete_images(request: DeleteImagesRequest):
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    failed = []
    for url in request.urls:
        # Extract the S3 key from URL
        parsed_url = urlparse(url)
        # Remove the leading '/' if exists
        s3_key = parsed_url.path.lstrip('/')
        
        success = delete_file_from_s3(s3_key)
        if not success:
            failed.append(url)

    if failed:
        return {
            "detail": "Some images could not be deleted",
            "failed_urls": failed
        }

    return {"detail": "All images deleted successfully"}
