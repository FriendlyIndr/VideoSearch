import json
from youtube_transcript_api import YouTubeTranscriptApi
import time
import random

def get_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id=video_id, languages=['hi'])
        print(transcript[:2])
        text = " ".join([t.text for t in transcript])
        return text
    except Exception as e:
        print(e)
        return None
    
# Load existing videos
with open("data/videos.json", "r", encoding="utf-8") as f:
    videos = json.load(f)

# Process each video
for video in videos:
    if "transcript" not in video or not video["transcript"]:
        print(f"Fetching transcript for {video['video_id']}...")

        transcript = get_transcript(video["video_id"])

        if transcript:
            video["transcript"] = transcript
        else:
            video["transcript"] = ""

        time.sleep(random.uniform(3, 7))

# Save updated data
with open("data/videos.json", "w", encoding="utf-8") as f:
    json.dump(videos, f, indent=2, ensure_ascii=False)

print("Done!")