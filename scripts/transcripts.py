from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound
)

def get_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()

        transcript = api.fetch(video_id=video_id, languages=['hi', 'en'])

        print(transcript[:2])
        text = " ".join([t.text for t in transcript])
        return {"status": "done", "text": text}
    except TranscriptsDisabled:
        print(f"Skipping {video_id} (subtitles disabled)")
        return {"status": "disabled", "text": None}
    except NoTranscriptFound:
        print(f"Skipping {video_id} (no transcript found)")
        return {"status": "not_found", "text": None}
    except Exception as e:
        print(e)
        return {"status": "error", "text": None}

if __name__ == "__main__": 
    import json
    import time
    import random

    # Load existing videos
    with open("data/videos.json", "r", encoding="utf-8") as f:
        videos = json.load(f)

    # Process each video
    for video in videos:
        if video.get("transcript_status") in ["done", "disabled", "not_found"]:
            continue
        
        print(f"Fetching transcript for {video['video_id']}...")

        result = get_transcript(video["video_id"])

        video["transcript"] = result["text"]
        video["transcript_status"] = result["status"]

        time.sleep(random.uniform(3, 7))

    # Save updated data
    with open("data/videos.json", "w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)

    print("Done!")