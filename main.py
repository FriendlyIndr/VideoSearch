from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle

from scripts import embeddings

app = FastAPI(title="Video Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("data/video_embeddings.pkl", "rb") as f:
    videos = pickle.load(f)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query: str, top_k: int = 5):
    query_embedding = embeddings.get_embedding(query)

    results = []
    for video in videos:
        for chunk in video.get("chunks", []):
            score = cosine_similarity(query_embedding, chunk["embedding"])
            results.append({
                "score": score, 
                "video_id": video["video_id"],
                "text": chunk["text"],
                "url": f"https://www.youtube.com/watch?v={video['video_id']}",
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


@app.get("/search")
def search_endpoint(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results"),
):
    results = search(q, top_k)
    return {"query": q, "results": results}

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html") as f:
        return f.read()