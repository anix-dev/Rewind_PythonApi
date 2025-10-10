from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.db.s3_client import upload_file_to_s3

router = APIRouter(tags=["Uploads"])

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
