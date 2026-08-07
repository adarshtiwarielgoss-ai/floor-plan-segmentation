from floor_segmentation.utils.common import configure_ultralytics

configure_ultralytics()
from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from floor_segmentation.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI(

    title="Floor Plan Segmentation API",

    description="YOLO26 Semantic Segmentation",

    version="1.0.0"

)


UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(exist_ok=True)


predictor = PredictionPipeline()


@app.get("/")
def home():

    return {

        "message": "Floor Plan Segmentation API",

        "status": "Running"

    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        file_path = UPLOAD_DIR / file.filename

        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(file.file, buffer)

        result = predictor.predict(str(file_path))

        return JSONResponse(result)

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )