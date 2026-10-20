import base64
import json
import os
import re
import smtplib
import socket
import ssl
import time
from collections import defaultdict
from email.message import EmailMessage

import certifi
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB uploads

ROBOFLOW_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL = os.getenv("ROBOFLOW_MODEL", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DETECTOR = os.getenv(
    "DETECTOR", "gemini" if GEMINI_KEY else ("claude" if ANTHROPIC_KEY else "roboflow")
).lower()
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
DEFAULT_AUTHORITY = os.getenv("AUTHORITY_EMAIL", "")
MAX_REPORTS_PER_HOUR = int(os.getenv("MAX_REPORTS_PER_HOUR", "5"))

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
sent_log = defaultdict(list)  # ip -> timestamps of sent reports


def too_many_reports(ip):
    now = time.time()
    sent_log[ip] = [t for t in sent_log[ip] if now - t < 3600]
    return len(sent_log[ip]) >= MAX_REPORTS_PER_HOUR


@app.get("/")
def home():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


PROMPT = (
    "You inspect road photos for potholes. A pothole is a bowl-shaped hole or broken "
    "section in the road surface. Ignore shadows, patches, manhole covers, road markings "
    "and puddles with no visible damage. Reply with JSON only, no other text, in this form: "
    '{"potholes":[{"box":[x1,y1,x2,y2],"confidence":0.0}]} where the box corners are '
    "fractions of the image width and height between 0 and 1 (x1,y1 is the top-left corner) "
    'and confidence is your certainty between 0 and 1. Use {"potholes":[]} if there are none.'
)


def detect_with_claude(photo_bytes, width, height):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CLAUDE_MODEL,
            "max_tokens": 600,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": base64.b64encode(photo_bytes).decode("ascii")}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        },
        timeout=60,
    )
    r.raise_for_status()
    return parse_claude_reply(r.json()["content"][0]["text"], width, height)


def parse_claude_reply(text, width, height):
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    preds = []
    for p in data.get("potholes", []):
        x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in p["box"]]
        if x2 <= x1 or y2 <= y1:
            continue
        preds.append({
            "x": (x1 + x2) / 2 * width,
            "y": (y1 + y2) / 2 * height,
            "width": (x2 - x1) * width,
            "height": (y2 - y1) * height,
            "confidence": max(0.0, min(1.0, float(p.get("confidence", 0.7)))),
        })
    return preds


GEMINI_PROMPT = (
    "You inspect road photos for potholes. A pothole is a bowl-shaped hole or broken "
    "section in the road surface. Ignore shadows, patches, manhole covers, road markings "
    "and puddles with no visible damage. Reply with JSON only in this form: "
    '{"potholes":[{"box_2d":[ymin,xmin,ymax,xmax],"confidence":0.0}]} where box_2d values '
    "are integers from 0 to 1000 relative to the image size, and confidence is your "
    'certainty between 0 and 1. Use {"potholes":[]} if there are none.'
)


def detect_with_gemini(photo_bytes, width, height):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": GEMINI_KEY, "content-type": "application/json"},
        json={
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg",
                                     "data": base64.b64encode(photo_bytes).decode("ascii")}},
                    {"text": GEMINI_PROMPT},
                ]
            }],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
        },
        timeout=120,
    )
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return parse_gemini_reply(text, width, height)


def parse_gemini_reply(text, width, height):
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    preds = []
    for p in data.get("potholes", []):
        ymin, xmin, ymax, xmax = [max(0.0, min(1000.0, float(v))) / 1000 for v in p["box_2d"]]
        if xmax <= xmin or ymax <= ymin:
            continue
        preds.append({
            "x": (xmin + xmax) / 2 * width,
            "y": (ymin + ymax) / 2 * height,
            "width": (xmax - xmin) * width,
            "height": (ymax - ymin) * height,
            "confidence": max(0.0, min(1.0, float(p.get("confidence", 0.7)))),
        })
    return preds


def detect_with_roboflow(photo_bytes, confidence):
    r = requests.post(
        f"https://detect.roboflow.com/{ROBOFLOW_MODEL}",
        params={"api_key": ROBOFLOW_KEY, "confidence": confidence},
        data=base64.b64encode(photo_bytes).decode("ascii"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("predictions", [])


@app.post("/api/detect")
def detect():
    photo = request.files.get("image")
    if not photo:
        return jsonify(error="No image received."), 400
    photo_bytes = photo.read()
    try:
        confidence = max(10, min(95, int(request.form.get("confidence", 40))))
        width = int(request.form.get("width", 0))
        height = int(request.form.get("height", 0))
    except ValueError:
        return jsonify(error="Invalid form values."), 400

    try:
        if DETECTOR == "gemini":
            if not GEMINI_KEY:
                return jsonify(error="GEMINI_API_KEY is missing in .env."), 500
            if width <= 0 or height <= 0:
                return jsonify(error="Image size missing."), 400
            preds = detect_with_gemini(photo_bytes, width, height)
        elif DETECTOR == "claude":
            if not ANTHROPIC_KEY:
                return jsonify(error="ANTHROPIC_API_KEY is missing in .env."), 500
            if width <= 0 or height <= 0:
                return jsonify(error="Image size missing."), 400
            preds = detect_with_claude(photo_bytes, width, height)
        else:
            if not ROBOFLOW_KEY or not ROBOFLOW_MODEL:
                return jsonify(error="Roboflow key or model is missing in .env."), 500
            preds = detect_with_roboflow(photo_bytes, confidence)
    except requests.HTTPError as exc:
        return jsonify(error=f"Detection service returned {exc.response.status_code}: {exc.response.text[:200]}"), 502
    except requests.Timeout:
        return jsonify(error="The detection service took too long. Check your internet and try again."), 504
    except requests.ConnectionError:
        return jsonify(error="Could not reach the detection service. Check your internet connection."), 502
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        return jsonify(error=f"Detection failed: {exc}"), 502
    return jsonify(predictions=preds)


@app.post("/api/report")
def report():
    if not SMTP_USER or not SMTP_PASSWORD:
        return jsonify(error="Email is not configured on the server."), 500

    ip = request.remote_addr or "unknown"
    if too_many_reports(ip):
        return jsonify(error="Report limit reached. Please try again in an hour."), 429

    to = request.form.get("to", "").strip() or DEFAULT_AUTHORITY
    reporter = request.form.get("reporter", "").strip()
    subject = " ".join(request.form.get("subject", "Pothole complaint").split())[:150]
    body = request.form.get("body", "").strip()
    photo = request.files.get("image")

    if not EMAIL_RE.match(to):
        return jsonify(error="Enter a valid authority email address."), 400
    if reporter and not EMAIL_RE.match(reporter):
        return jsonify(error="Your email address looks invalid."), 400
    if not body:
        return jsonify(error="The complaint text is empty."), 400
    if not photo:
        return jsonify(error="Add a photo before sending."), 400

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"Pothole Reporter <{SMTP_USER}>"
    msg["To"] = to
    if reporter:
        msg["Reply-To"] = reporter
        msg["Cc"] = reporter  # the reporter gets a copy as proof
    msg.set_content(body)
    msg.add_attachment(
        photo.read(), maintype="image", subtype="jpeg", filename="pothole-report.jpg"
    )

    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(cafile=certifi.where())) as s:
                s.login(SMTP_USER, SMTP_PASSWORD)
                s.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.starttls(context=ssl.create_default_context(cafile=certifi.where()))
                s.login(SMTP_USER, SMTP_PASSWORD)
                s.send_message(msg)
    except socket.gaierror:
        return jsonify(error=f"Cannot find the mail server '{SMTP_HOST}'. Check SMTP_HOST in .env and your internet connection."), 502
    except smtplib.SMTPAuthenticationError:
        return jsonify(error="Gmail rejected the login. Check SMTP_USER and the App Password in .env."), 502
    except (smtplib.SMTPException, OSError) as exc:
        return jsonify(error=f"Could not send the email: {exc}"), 502

    sent_log[ip].append(time.time())
    return jsonify(ok=True, sent_to=to)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
