import json

with open("data/videos.json", "r", encoding="utf-8") as f:
    videos = json.load(f)

def add_video(videos, new_video):
    existing_videos = {v["video_id"] for v in videos}

    if new_video["video_id"] not in existing_videos:
        videos.append(new_video)
    else:
        print(f"Duplicate skipped: {new_video['video_id']}")

new_video = {
    "video_id": "jc7vCNvC-pU",
    "title": "he said WHAAT"
}
add_video(videos, new_video)

# Save changes
with open("data/videos.json", "w", encoding="utf-8") as f:
    json.dump(videos, f, indent=2, ensure_ascii=False)