from sentence_transformers import SentenceTransformer
import json
import pickle

print('Fetching model')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_embedding(text):
    return model.encode(text)

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

if __name__ == "__main__":
    # Load existing videos
    with open("data/video_embeddings.pkl", "rb") as f:
        videos = pickle.load(f)

    # For each video
    for video in videos:
        if "transcript" in video and video["transcript"]:

            # Skip if already embedded
            if "chunks" in video and video["chunks"]:
                continue

            chunks = chunk_text(video["transcript"])
            video["chunks"] = []

            for chunk in chunks:
                embedding = get_embedding(chunk)

                video["chunks"].append({
                    "text": chunk,
                    "embedding": embedding.tolist()
                })

    with open("data/video_embeddings.pkl", "wb") as f:
        pickle.dump(videos, f)