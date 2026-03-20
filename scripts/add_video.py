import json
import sys
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    
    return None

def add_video(video_id):
    with open("data/videos.json", "r", encoding="utf-8") as f:
        videos = json.load(f)

    existing_ids = {v["video_id"] for v in videos}

    if video_id in existing_ids:
        print(f"Duplicate skipped: {video_id}")
        return
        
    new_video = {
        "video_id": video_id,
        "title": ""
    }

    videos.append(new_video)

    # Save changes
    with open("data/videos.json", "w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)

    print(f"Added video: {video_id}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python add_video.py <youtube_url>")
        sys.exit(1)

    url = sys.argv[1]
    video_id = extract_video_id(url)

    if not video_id:
        print(f"Invalid Youtube URL")
        sys.exit(1)
    
    add_video(video_id)