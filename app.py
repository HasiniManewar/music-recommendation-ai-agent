import os
import re
import csv
import io
import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from flask import Flask, request, jsonify, render_template_string
from google import genai
from google.genai import types

app = Flask(__name__)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = "gemini-3.7-flash"

DATASET_BASE = "https://raw.githubusercontent.com/harin-git/GlobalMood/main/Data/"
SONG_METADATA_URL = DATASET_BASE + "songmeta_GlobalMood.csv"
MOOD_DATA_URL = DATASET_BASE + "chains_GlobalMood.csv"

MAX_INPUT_LENGTH = 500
DEFAULT_LIMIT = 5
MAX_LIMIT = 10

UNRELATED_MESSAGE = (
    "I'm a Music Recommendation Agent. I can help you find songs and artists "
    "based on your preferred type or mood of music."
)

HARMFUL_TERMS = {
    "terrorist", "terrorism", "bomb making", "weapon making",
    "how to kill", "how to murder", "suicide method",
    "self harm method", "child sexual", "sexual abuse", "rape instructions",
}

MUSIC_KEYWORDS = {
    "song", "songs", "music", "artist", "singer", "album", "track",
    "melody", "melodic", "genre", "style", "mood", "feeling", "playlist",
    "listen", "soothing", "sad", "happy", "romantic", "relaxing",
    "energetic", "motivational", "calm", "peaceful", "love", "chill",
    "excited", "emotional", "dance", "rock", "pop", "jazz", "classical",
    "hip hop", "rap", "folk", "country", "indie", "metal", "electronic",
}

DATA_CACHE: Dict[str, Any] = {"songs": None, "moods": None}

gemini_client = None
if GOOGLE_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception:
        gemini_client = None


def normalize(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip().lower()
    return re.sub(r"\s+", " ", value)


def safe_int(value: Any, default: int = DEFAULT_LIMIT) -> int:
    try:
        return max(1, min(int(value), MAX_LIMIT))
    except (TypeError, ValueError):
        return default


def extract_requested_limit(text: str) -> int:
    match = re.search(
        r"\b(?:give|suggest|recommend|show|find)\s+(?:me\s+)?(\d{1,2})\b",
        text, re.I
    )
    if not match:
        match = re.search(r"\b(\d{1,2})\s+(?:songs?|tracks?)\b", text, re.I)
    return safe_int(match.group(1)) if match else DEFAULT_LIMIT


def is_harmful(text: str) -> bool:
    normalized = normalize(text)
    return any(term in normalized for term in HARMFUL_TERMS)


def looks_music_related(text: str) -> bool:
    normalized = normalize(text)
    if any(keyword in normalized for keyword in MUSIC_KEYWORDS):
        return True
    if re.search(r"\b(?:who|which)\b.*\b(?:artist|singer)\b", normalized):
        return True
    if re.search(r"\b(?:artist|singer)\b.*\bof\b", normalized):
        return True
    return False


def extract_song_name(text: str) -> Optional[str]:
    patterns = [
        r"(?:artist|singer)\s+(?:of|for)\s+[\"']?(.+?)[\"']?\s*$",
        r"(?:who\s+(?:is|was)\s+(?:the\s+)?(?:artist|singer)\s+(?:of|for))\s+[\"']?(.+?)[\"']?\s*$",
        r"(?:who\s+(?:sang|sings))\s+[\"']?(.+?)[\"']?\s*$",
        r"(?:artist|singer)\s*:\s*[\"']?(.+?)[\"']?\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip(), re.I)
        if match:
            song = match.group(1).strip(" \"'?.!")
            if song:
                return song
    return None


def detect_artist_request(text: str) -> bool:
    normalized = normalize(text)
    return bool(
        re.search(r"\b(?:artist|singer)\b", normalized)
        and ("of " in normalized or "for " in normalized or
             "who" in normalized or "sang" in normalized)
    )


def fetch_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Music-Recommendation-AI-Agent/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8-sig", errors="replace")


def find_column(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    normalized_fields = {
        normalize(field).replace("_", "").replace("-", "").replace(" ", ""): field
        for field in fieldnames if field
    }
    for candidate in candidates:
        key = normalize(candidate).replace("_", "").replace("-", "").replace(" ", "")
        if key in normalized_fields:
            return normalized_fields[key]
    for normalized_name, original in normalized_fields.items():
        for candidate in candidates:
            candidate_key = normalize(candidate).replace("_", "").replace("-", "").replace(" ", "")
            if candidate_key in normalized_name or normalized_name in candidate_key:
                return original
    return None


def load_song_metadata() -> List[Dict[str, str]]:
    if DATA_CACHE["songs"] is not None:
        return DATA_CACHE["songs"]

    content = fetch_text(SONG_METADATA_URL)
    reader = csv.DictReader(io.StringIO(content))
    rows = []

    for row in reader:
        song = row.get("song", "").strip()
        artist = row.get("artist", "").strip()
        country = row.get("country", "").strip()
        if song:
            rows.append({"song": song, "artist": artist, "country": country})

    DATA_CACHE["songs"] = rows
    return rows


def load_mood_data() -> List[Dict[str, str]]:
    if DATA_CACHE["moods"] is not None:
        return DATA_CACHE["moods"]

    content = fetch_text(MOOD_DATA_URL)
    reader = csv.DictReader(io.StringIO(content))

    if not reader.fieldnames:
        DATA_CACHE["moods"] = []
        return []

    fields = list(reader.fieldnames)
    song_column = find_column(fields, [
        "song", "song_name", "songname", "title", "track",
        "track_name", "trackname", "videoID", "video_id"
    ])
    tag_column = find_column(fields, [
        "tag", "mood", "mood_tag", "moodtag", "descriptor",
        "description", "word", "chain"
    ])

    if not song_column or not tag_column:
        DATA_CACHE["moods"] = []
        return []

    rows = []
    for row in reader:
        song = str(row.get(song_column, "")).strip()
        tag = str(row.get(tag_column, "")).strip()
        if song and tag:
            rows.append({"song": song, "tag": tag})

    DATA_CACHE["moods"] = rows
    return rows


def music_tool(music_request: str, limit: int = DEFAULT_LIMIT) -> List[Dict[str, str]]:
    """Search only the GitHub dataset for matching songs."""
    query = normalize(music_request)
    if not query:
        return []

    limit = safe_int(limit)
    songs = load_song_metadata()
    mood_rows = load_mood_data()

    song_index = {}
    for song in songs:
        key = normalize(song["song"])
        if key and key not in song_index:
            song_index[key] = song

    matching_song_names = set()
    query_words = [
        word for word in re.findall(r"[a-zA-ZÀ-ÿ0-9']+", query)
        if len(word) > 2
    ]

    for row in mood_rows:
        tag = normalize(row["tag"])
        song_name = row["song"].strip()
        if not tag or not song_name:
            continue
        if query in tag or any(word in tag for word in query_words):
            matching_song_names.add(normalize(song_name))

    results = []
    for key, song in song_index.items():
        if key in matching_song_names:
            results.append({
                "song": song["song"],
                "artist": song["artist"],
                "country": song["country"],
            })
        if len(results) >= limit:
            break

    if not results:
        for song in songs:
            song_name = normalize(song["song"])
            if query == song_name or query in song_name:
                results.append({
                    "song": song["song"],
                    "artist": song["artist"],
                    "country": song["country"],
                })
            if len(results) >= limit:
                break

    unique = []
    seen = set()
    for result in results:
        key = normalize(result["song"])
        if key not in seen:
            seen.add(key)
            unique.append(result)

    return unique[:limit]


def artist_tool(song_name: str) -> Optional[Dict[str, str]]:
    """Look up an artist only in the GitHub dataset."""
    if not song_name:
        return None

    songs = load_song_metadata()
    requested = normalize(song_name)

    for song in songs:
        if normalize(song["song"]) == requested:
            return {
                "song": song["song"],
                "artist": song["artist"],
                "available": bool(song["artist"].strip()),
            }

    possible_matches = [
        song for song in songs
        if requested in normalize(song["song"]) or
           normalize(song["song"]) in requested
    ]

    if len(possible_matches) == 1:
        song = possible_matches[0]
        return {
            "song": song["song"],
            "artist": song["artist"],
            "available": bool(song["artist"].strip()),
        }

    return None


def deterministic_response(tool_results: Any, response_type: str) -> str:
    if response_type == "artist":
        if not tool_results:
            return "I couldn't find that song in the available dataset."
        if not tool_results.get("available"):
            return (
                f"'{tool_results.get('song', 'That song')}' exists in the dataset, "
                "but the artist information is unavailable."
            )
        return f"🎵 {tool_results['song']}\n👤 Artist: {tool_results['artist']}"

    if not tool_results:
        return "No matching songs were found in the available dataset."

    lines = ["🎵 Here are some songs from the dataset:\n"]
    for item in tool_results:
        if item.get("artist"):
            lines.append(f"• {item['song']} — {item['artist']}")
        else:
            lines.append(f"• {item['song']}")
    return "\n".join(lines)


def generate_ai_response(
    user_request: str,
    tool_results: Any,
    response_type: str = "recommendation",
) -> str:
    if not gemini_client:
        return deterministic_response(tool_results, response_type)

    if response_type == "artist":
        instruction = """
You are a concise music assistant.
Use ONLY the verified dataset result below.
Never invent or guess an artist.
If unavailable, say artist information is unavailable.
Keep the answer short and friendly.
"""
    else:
        instruction = """
You are a concise music recommendation assistant.
Use ONLY the verified songs and artists below.
Never add or invent songs or artists.
If the list is empty, say no matching songs were found.
Keep the answer short and friendly.
"""

    prompt = (
        instruction
        + "\n\nUser request:\n"
        + user_request
        + "\n\nVerified dataset results:\n"
        + json.dumps(tool_results, ensure_ascii=False)
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=300,
            ),
        )
        text = (response.text or "").strip()
        if text:
            return text
    except Exception:
        pass

    return deterministic_response(tool_results, response_type)


def process_request(user_request: str) -> str:
    user_request = user_request.strip()

    if not user_request:
        return "Please enter a music request, such as 'Give me some sad songs.'"

    if len(user_request) > MAX_INPUT_LENGTH:
        return f"Please keep your request under {MAX_INPUT_LENGTH} characters."

    if is_harmful(user_request):
        return (
            "I can't help with harmful, illegal, hateful, or sexually explicit "
            "requests. I can help you discover music instead."
        )

    if not looks_music_related(user_request):
        return UNRELATED_MESSAGE

    if detect_artist_request(user_request):
        song_name = extract_song_name(user_request)
        if not song_name:
            return (
                "Please tell me the song name, for example: "
                "\"Who is the artist of Perfect?\""
            )
        try:
            result = artist_tool(song_name)
            if not result:
                return (
                    f"I couldn't find '{song_name}' in the available dataset, "
                    "so I can't provide an artist."
                )
            return generate_ai_response(user_request, result, "artist")
        except urllib.error.URLError:
            return "I couldn't access the GitHub music dataset right now. Please try again."
        except Exception:
            return "Something went wrong while searching the music dataset. Please try again."

    limit = extract_requested_limit(user_request)

    try:
        results = music_tool(user_request, limit)
        return generate_ai_response(user_request, results, "recommendation")
    except urllib.error.URLError:
        return "I couldn't access the GitHub music dataset right now. Please try again."
    except Exception:
        return "Something went wrong while searching the music dataset. Please try again."


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Music Recommendation AI Agent</title>
<style>
*{box-sizing:border-box}
body{
    margin:0;min-height:100vh;font-family:Arial,sans-serif;color:white;
    display:flex;align-items:center;justify-content:center;padding:20px;
    background:radial-gradient(circle at top left,#6d28d9,transparent 35%),
               radial-gradient(circle at bottom right,#db2777,transparent 35%),
               #111827;
}
.container{
    width:100%;max-width:760px;background:rgba(17,24,39,.9);
    border:1px solid rgba(255,255,255,.12);border-radius:24px;
    padding:32px;box-shadow:0 25px 70px rgba(0,0,0,.35)
}
.logo{text-align:center;font-size:52px;margin-bottom:8px}
h1{text-align:center;margin:0;font-size:32px}
.subtitle{text-align:center;color:#cbd5e1;margin:12px 0 28px;line-height:1.5}
.examples{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:22px}
.example{
    border:1px solid rgba(255,255,255,.15);background:rgba(255,255,255,.07);
    color:#e5e7eb;padding:8px 12px;border-radius:999px;cursor:pointer;font-size:13px
}
.input-area{display:flex;gap:10px}
input{
    flex:1;min-width:0;border:none;outline:none;padding:16px;border-radius:14px;
    background:#f8fafc;color:#111827;font-size:16px
}
button{
    border:none;border-radius:14px;padding:0 24px;background:#ec4899;color:white;
    font-weight:bold;font-size:15px;cursor:pointer
}
button:hover{background:#db2777}
button:disabled{opacity:.6;cursor:not-allowed}
.loading{display:none;text-align:center;color:#cbd5e1;margin-top:18px}
.answer{
    display:none;margin-top:24px;padding:22px;border-radius:18px;
    background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);
    line-height:1.65;white-space:pre-wrap;word-wrap:break-word
}
.footer{text-align:center;color:#94a3b8;font-size:12px;margin-top:24px}
@media(max-width:600px){
    .container{padding:22px}h1{font-size:25px}.input-area{flex-direction:column}
    button{height:52px}
}
</style>
</head>
<body>
<main class="container">
<div class="logo">🎵</div>
<h1>Music Recommendation AI Agent</h1>
<p class="subtitle">
Discover songs based on your mood, feeling, music type, genre, style,
or artist-related requests.
</p>

<div class="examples">
<button class="example" type="button" onclick="setExample('Give me some sad songs')">Sad</button>
<button class="example" type="button" onclick="setExample('I want soothing music')">Soothing</button>
<button class="example" type="button" onclick="setExample('Give me 5 happy songs')">Happy</button>
<button class="example" type="button" onclick="setExample('Give me romantic songs')">Romantic</button>
<button class="example" type="button" onclick="setExample('Give me energetic music')">Energetic</button>
</div>

<form id="musicForm">
<div class="input-area">
<input id="requestInput" type="text" maxlength="500"
autocomplete="off" placeholder="What kind of music are you looking for?" required>
<button id="askButton" type="submit">Ask 🎧</button>
</div>
</form>

<div id="loading" class="loading">🎶 Searching the music dataset...</div>
<div id="answer" class="answer"></div>

<div class="footer">
Recommendations are based only on songs available in the connected music dataset.
</div>
</main>

<script>
const form=document.getElementById("musicForm");
const input=document.getElementById("requestInput");
const button=document.getElementById("askButton");
const loading=document.getElementById("loading");
const answer=document.getElementById("answer");

function setExample(text){input.value=text;input.focus()}

form.addEventListener("submit",async function(event){
    event.preventDefault();
    const text=input.value.trim();

    if(!text){
        answer.style.display="block";
        answer.textContent="Please enter a music request.";
        return;
    }

    button.disabled=true;
    loading.style.display="block";
    answer.style.display="none";

    try{
        const response=await fetch("/api/ask",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({request:text})
        });
        const data=await response.json();
        answer.style.display="block";
        answer.textContent=data.answer || "Something went wrong. Please try again.";
    }catch(error){
        answer.style.display="block";
        answer.textContent="Unable to connect to the Music Recommendation Agent.";
    }finally{
        button.disabled=false;
        loading.style.display="none";
    }
});
</script>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(PAGE)


@app.post("/api/ask")
def ask():
    try:
        data = request.get_json(silent=True) or {}
        user_request = data.get("request", "")

        if not isinstance(user_request, str):
            return jsonify({"answer": "Please enter a valid text music request."}), 400

        user_request = user_request.strip()

        if not user_request:
            return jsonify({"answer": "Please enter a music request."}), 400

        if len(user_request) > MAX_INPUT_LENGTH:
            return jsonify({
                "answer": f"Please keep your request under {MAX_INPUT_LENGTH} characters."
            }), 400

        return jsonify({"answer": process_request(user_request)})

    except Exception:
        return jsonify({
            "answer": "Something went wrong. Please try your music request again."
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
