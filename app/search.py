import numpy as np
from scripts import embeddings
import pickle

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load existing videos
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

results = search("कुछ न होने से कुछ होना बेहतर है। अपूर्ण प्रगति भी प्रगति ही है।", videos)

for r in results:
    print(f"Score: {r['score']:.4f}")
    print(f"Video ID: {r['video_id']}")
    print(f"Text: {r['text'][:200]}")
    print("-" * 50)