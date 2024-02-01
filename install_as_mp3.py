import json
from pytube import YouTube
from googleapiclient.discovery import build

# Load video data from the JSON file videoid_title
with open("videoid_title.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Loop through each video in the data
for video in data:
    video_id = video["video_id"]
    title = video["title"]
    
    # Construct the YouTube URL
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Create a YouTube object
    yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)

    # Choose the audio stream (highest bitrate by default)
    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
    title = yt.title.replace('|', '_').replace("-","_").replace("&","").replace("-","").replace("?","").replace("!","").replace(".","").replace("/","")
    # Set the download filename (including extension)
    filename = f"mp3_files\{title}.mp3"

    # Download the audio stream and save it as the chosen filename
    audio_stream.download(filename=filename)

    print(f"Downloaded: {title}")
