from fastapi import FastAPI,File, UploadFile
from app.model.model import __version__ as model_version
from app.model.model import predict_pipeline
from pydantic import BaseModel
from PIL import Image
import io

app = FastAPI()

class PredictionOut(BaseModel):
    prediction: str

model_version = model_version

@app.get("/")
def home():
    return {'check':'OK','model_version':model_version}

@app.post("/predict", response_model=PredictionOut)
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    prediction = predict_pipeline(image)
    return {"caption": prediction}