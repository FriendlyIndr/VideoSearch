import numpy as np
from scripts import embeddings
import pickle
import sys

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load existing videos' embeddings
with open("data/video_embeddings.pkl", "rb") as f:
    videos = pickle.load(f)

def search(query, videos):
    query_embedding = embeddings.get_embedding(query)

    results = []
    for video in videos:
        for chunk in video.get("chunks", []):
            score = cosine_similarity(query_embedding, chunk["embedding"])
            results.append({
                "score": score, 
                "video_id": video["video_id"],
                "text": chunk["text"]
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:3]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python -m app.search \"your query here\"")
        sys.exit(1)

    # Join all words into a single query
    query = " ".join(sys.argv[1:])
    
    results = search(query, videos)

    for r in results:
        print(f"Score: {r['score']:.4f}")
        print(f"Video URL: https://www.youtube.com/watch?v={r["video_id"]}")
        print(f"Text: {r['text'][:200]}")
        print("-" * 50)