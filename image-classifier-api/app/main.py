from fastapi import FastAPI, UploadFile, File, HTTPException
from app.schema import ClassificationOutput
from app import model

app = FastAPI(
    title="Image Classifier API",
    description="Upload any image, get top-5 ImageNet predictions",
    version="3.0.0"
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "ResNet50"}

@app.post("/classify", response_model=ClassificationOutput)
async def classify(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: jpeg, png, webp"
        )

    # Read raw bytes
    image_bytes = await file.read()

    # Validate file is not empty
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    predictions = model.predict(image_bytes, top_k=5)

    return ClassificationOutput(top_predictions=predictions)