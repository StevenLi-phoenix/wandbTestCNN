import io
from contextlib import asynccontextmanager
from pathlib import Path

import fastapi
import torch
import uvicorn
from fastapi import File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from predict import Predictor


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    app.state.predictor = Predictor()
    yield


app = fastapi.FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = Path(__file__).parent / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict(request: Request, image: UploadFile = File(...)):
    if image.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/bmp"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    predictor: Predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor is not initialized")

    logits = predictor.predict_file(io.BytesIO(contents)).squeeze(0).cpu()
    probabilities = torch.softmax(logits, dim=0)
    predicted_class = int(probabilities.argmax().item())

    return {
        "prediction": predicted_class,
        "probabilities": probabilities.tolist(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
