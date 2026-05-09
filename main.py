from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle
import json
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs
import uuid

print("Starting app...")
from scripts.add_video import get_title
from scripts.transcripts import get_transcript
from scripts import embeddings
print("Embeddings imported")

app = FastAPI(title="Video Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

with open("data/video_embeddings.pkl", "rb") as f:
    videos = pickle.load(f)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def keyword_score(query, text):
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())

    overlap = query_words.intersection(text_words)
    return len(overlap) / len(query_words) # simple ratio

def search(query: str, top_k: int = 5):
    query_embedding = embeddings.get_embedding(query)

    results = []
    for video in videos:
        # Search transcript chunks
        for chunk in video.get("chunks", []):
            score = cosine_similarity(query_embedding, chunk["embedding"])
            results.append({
                "score": score, 
                "video_id": video["video_id"],
                "text": chunk["text"],
                "type": "transcript",
                "url": f"https://www.youtube.com/watch?v={video['video_id']}",
            })

        # Search explanations
        for exp in video.get("explanations", []):
            semantic_score = cosine_similarity(query_embedding, exp["embedding"])
            keyword_boost = keyword_score(query, exp["text"])

            final_score = semantic_score + 0.3 * keyword_boost # tune 0.3

            results.append({
                "score": final_score,
                "video_id": video["video_id"],
                "text": exp["text"],
                "type": "explanation",
                "url": f"https://www.youtube.com/watch?v={video['video_id']}",
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]

class IngestRequest(BaseModel):
    url: str

def extract_video_id(url: str):
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]
        
        if parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/shorts/")[1]
    
    return None

@app.post("/ingest")
def ingest_endpoint(body: IngestRequest):
    # Extract video id from url
    video_id = extract_video_id(body.url)

    if not video_id:
        return {"error": "Invalid YouTube URL"}, 400
    
    #Check for duplicates in the in-memory list
    existing_ids = {v["video_id"] for v in videos}
    if video_id in existing_ids:
        return {"error": "Video already ingested", "video_id": video_id}
    
    # Build the video entry
    new_video = {
        "video_id": video_id,
        "title": get_title(video_id),  # reuse your existing helper
    }

    # Fetch transcript
    result = get_transcript(video_id)  # reuse your existing helper
    new_video["transcript"] = result["text"]
    new_video["transcript_status"] = result["status"]

    # Generate embeddings if transcript is available
    if result["text"]:
        chunks = embeddings.chunk_text(result["text"])
        new_video["chunks"] = []
        for chunk in chunks:
            embedding = embeddings.get_embedding(chunk)
            new_video["chunks"].append({
                "text": chunk,
                "embedding": embedding.tolist(),
            })
    else:
        new_video["chunks"] = []

    # Persist to both files
    videos.append(new_video)

    with open("data/videos.json", "r+", encoding="utf-8") as f:
        existing = json.load(f)
        existing.append({k: v for k, v in new_video.items() if k != "chunks"})
        f.seek(0)
        json.dump(existing, f, indent=2, ensure_ascii=False)

    with open("data/video_embeddings.pkl", "wb") as f:
        pickle.dump(videos, f)

    return {
        "video_id": video_id,
        "title": new_video["title"],
        "transcript_status": new_video["transcript_status"],
        "chunks_generated": len(new_video["chunks"]),
    }

class AddExplanationRequest(BaseModel):
    video_id: str
    text: str

@app.post("/add_explanation")
def add_explanation(body: AddExplanationRequest):
    #Find video
    video = next((v for v in videos if v["video_id"] == body.video_id), None)

    if not video:
        return {"error": "Video not found"}, 404

    #Generate embedding
    embedding = embeddings.get_embedding(body.text)

    #Initialize explanations if not present
    if "explanations" not in video:
        video["explanations"] = []

    #Add explanation
    video["explanations"].append({
        "id": str(uuid.uuid4()),
        "text": body.text,
        "embedding": embedding.tolist(),
    })

    #Persist the explanation
    with open("data/video_embeddings.pkl", "wb") as f:
        pickle.dump(videos, f)

    return {"message": "Explanation added successfully"}

class EditExplanationRequest(BaseModel):
    video_id: str
    explanation_id: str
    new_text: str

@app.post("/edit_explanation")
def edit_explanation(body: EditExplanationRequest):
    video = next((v for v in videos if v["video_id"] == body.video_id), None)

    if not video:
        return {"error": "Video not found"}, 404

    for exp in video.get("explanations", []):
        if exp["id"] == body.explanation_id:
            exp["text"] = body.new_text
            exp["embedding"] = embeddings.get_embedding(body.new_text).tolist()

            with open("data/video_embeddings.pkl", "wb") as f:
                pickle.dump(videos, f)

            return {"message": "Explanation updated"}
        
    return {"error": "Explanation not found"}, 404


@app.get("/videos")
def list_videos():
    return [
        {
            "video_id": v["video_id"], 
            "title": v.get("title", ""),
            "explanations": v.get("explanations", [])
        } for v in videos
    ]

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