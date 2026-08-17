"""
Demo API

Run:
  cd inference/api
  uvicorn main:app --reload --port 8000

Then open: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import os
from mangum import Mangum
from data_loader import DataLoader

# Paths
PRED_DIR = Path(os.getenv("PRED_DIR", "../../inference/predictions"))
TEST_IMG_DIR = Path(os.getenv("TEST_IMG_DIR", "../../data/memes/test/memes"))
TRAIN_IMG_DIR = Path(os.getenv("TRAIN_IMG_DIR", "../../data/memes/train/memes"))
TEST_PARQUET = Path(os.getenv("TEST_PARQUET", "../../data/processed/test_model_ready.parquet"))
TRAIN_PARQUET = Path(os.getenv("TRAIN_PARQUET", "../../data/processed/train_base.parquet"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", "static"))

dl: DataLoader = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global dl

    #Load Prediction Data
    print("Loading train and prediction data...")
    dl = DataLoader(pred_dir=PRED_DIR,test_parquet=TEST_PARQUET,train_parquet=TRAIN_PARQUET)

    print(f"    {len(dl.train_df)} train loaded")
    print(f"    {len(dl.df)} predictions loaded")
    print(f"    Models: {dl.available_models}")
    yield
    print("Shutting down.")

app = FastAPI(
    title="Multimodal Sexism Detection Demo",
    description="Demo API - Diego Blassio",
    version="1.0.0",
    lifespan=lifespan)

# Static File
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Routes
@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/explorer", include_in_schema=False)
async def explorer():
    return FileResponse(STATIC_DIR / "explorer.html")

@app.get("/gates", include_in_schema=False)
async def gates_page():
    return FileResponse(STATIC_DIR / "gates.html")

@app.get("/disagree", include_in_schema=False)
async def disagree_page():
    return FileResponse(STATIC_DIR / "disagree.html")

@app.get("/train", include_in_schema=False)
async def train_page():
    return FileResponse(STATIC_DIR / "train.html")


#Images Serving
@app.get("/images/{filename}", include_in_schema=False)
async def serve_image(filename: str):
    for img_dir in [TEST_IMG_DIR, TRAIN_IMG_DIR]:
        path = img_dir / filename
        if path.exists():
            return FileResponse(path)

    raise HTTPException(status_code=404, detail="Image not found")


# API Endpoints

#Glogal Stats
@app.get("/api/stats")
async def get_stats():
    """
    Global statistics per model.
    index.html dashboard
    """
    return JSONResponse(dl.get_stats())

#Train stats
@app.get("/api/train/stats")
async def train_stats():
    return JSONResponse(dl.get_train_stats())


@app.get("/api/train/memes")
async def train_memes(
    page: int = 1,
    page_size: int = 24,
    lang: Optional[str] = None,
    min_task21_soft: Optional[float] = 0,
    min_task22_soft: Optional[float] = 0,
    category: Optional[str] = None,
    search: Optional[str] = None):

    return JSONResponse(dl.get_train_memes(
        page=page,
        page_size=page_size,
        lang=lang,
        min_task21_soft=min_task21_soft,
        min_task22_soft=min_task22_soft,
        category=category,
        search=search))


@app.get("/api/train/memes/{meme_id}")
async def train_meme_detail(meme_id: str):
    detail = dl.get_train_meme_detail(meme_id)

    if detail is None:
        raise HTTPException(status_code=404, detail="Training meme not found")

    return detail


# Prediction Dataset

#Get memes
@app.get("/api/memes")
async def get_memes(
    page:int = Query(1,    ge=1),
    page_size:int = Query(20, ge=1, le=100),
    lang: Optional[str] = Query(None, description="'en' or 'es'"),
    prediction: Optional[str] = Query(None, description="'sexist' or 'not_sexist'"),
    model: Optional[str] = Query(None, description="Model to filter by prediction"),
    search: Optional[str] = Query(None, description="Text search in meme text"),
):
    """
    Paginated meme list with optional filters.
    Used by: explorer.html
    """
    results = dl.get_memes(
        page=page,
        page_size=page_size,
        lang=lang,
        prediction=prediction,
        model=model,
        search=search,
    )
    return JSONResponse(results)

# Per Meme Full Description and Prediction
@app.get("/api/memes/{meme_id}")
async def get_meme(meme_id: str):
    """
    Full prediction details for one meme (all models + gates).
    Used by: explorer.html detail panel, gates.html
    """
    meme = dl.get_meme_detail(meme_id)
    if meme is None:
        raise HTTPException(status_code=404, detail=f"Meme {meme_id} not found")
    return JSONResponse(meme)


# Show sorted by disagreement score 
@app.get("/api/disagreements")
async def get_disagreements(
    page:      int = Query(1,  ge=1),
    page_size: int = Query(20, ge=1, le=100),
    task:      str = Query("2.1", description="Task to check disagreement on: '2.1', '2.2', '2.3'"),
):
    """
    Memes where models disagree, sorted by disagreement score.
    Used by: disagree.html
    """
    results = dl.get_disagreements(page=page, page_size=page_size, task=task)
    return JSONResponse(results)

#Gates Endpoint
@app.get("/api/gates")
async def get_gates(
    page:      int = Query(1,      ge=1),
    page_size: int = Query(24,     ge=1, le=100),
    sort_by:   str = Query("beta", description="beta | alpha | lambda"),
):
    """Gate values per meme with distribution histograms. Used by: gates.html"""
    return JSONResponse(dl.get_gates_data(page=page, page_size=page_size, sort_by=sort_by))

#List of Models available 
@app.get("/api/models")
async def get_models():
    """List of available models."""
    return JSONResponse({"models": dl.available_models})


# Lambda Handler for AWS Lambda
handler = Mangum(app)
